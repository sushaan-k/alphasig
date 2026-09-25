"""Risk-factor diff throughput.

Two layers are timed:

* ``compute_text_similarity`` (the word-level ``difflib`` gate that decides
  whether the LLM is called at all) on section pairs of increasing size,
  including an unchanged section (the case that skips the LLM), and
* ``RiskDifferEngine.extract`` end to end with a zero-latency mock LLM, i.e.
  the CPU cost of diffing consecutive filings excluding model time.
"""

from __future__ import annotations

import asyncio
import itertools
import random
from datetime import date
from functools import partial
from typing import Any

from alphasig.engines.risk_differ import RiskDifferEngine, compute_text_similarity
from alphasig.models import FilingSection, FilingType
from benchmarks._common import SEED, timeit
from benchmarks.fixtures import evolve_risks, risk_paragraphs


class _ZeroLatencyLLM:
    """Stands in for :class:`alphasig.llm.LLMClient`; returns one change."""

    calls = 0

    async def extract_json(self, system: str, user: str, **_: Any) -> Any:
        self.calls += 1
        return [
            {
                "change_type": "ESCALATED",
                "risk": "Supply chain concentration",
                "language_shift": "'may' -> 'is currently'",
                "severity_estimate": "HIGH",
                "confidence": 0.8,
                "related_tickers": [],
            }
        ]


def _section(text: str, filed: date) -> FilingSection:
    return FilingSection(
        filing_accession=f"acc-{filed.isoformat()}",
        ticker="BENCH",
        section_name="Risk Factors",
        section_key="risk_factors",
        text=text,
        filing_type=FilingType.TEN_K,
        filed_date=filed,
    )


def _pair(n_paras: int, seed: int) -> tuple[str, str]:
    rng = random.Random(seed)
    prev = risk_paragraphs(rng, n_paras)
    cur = evolve_risks(rng, prev)
    return " ".join(prev), " ".join(cur)


def run(quick: bool = False) -> dict[str, Any]:
    repeat = 2 if quick else 3
    sizes = (10, 40) if quick else (10, 40, 80)
    sim_rows = []
    cases: list[tuple[str, int]] = [("evolved", n) for n in sizes]
    cases.append(("identical", 40))  # unchanged section: the LLM-skip case
    for kind, n in cases:
        prev, cur = _pair(n, SEED + n)
        if kind == "identical":
            cur = prev
        sim = compute_text_similarity(prev, cur)
        stats = timeit(partial(compute_text_similarity, prev, cur), repeat=repeat)
        chars = len(prev) + len(cur)
        sim_rows.append(
            {
                "pair": kind,
                "paragraphs": n,
                "chars_prev": len(prev),
                "chars_cur": len(cur),
                "similarity": round(sim, 4),
                "median_s": round(stats["median_s"], 4),
                "chars_per_s": round(chars / stats["median_s"]),
            }
        )

    # Engine throughput over a synthetic 10-K history (consecutive pairs).
    n_filings = 4 if quick else 8
    rng = random.Random(SEED)
    paras = risk_paragraphs(rng, 45)
    history = []
    for i in range(n_filings):
        history.append(_section(" ".join(paras), date(2005 + i, 2, 1)))
        paras = evolve_risks(rng, paras)
    engine = RiskDifferEngine()

    async def diff_all(llm: _ZeroLatencyLLM) -> int:
        count = 0
        for p_sec, c_sec in itertools.pairwise(history):
            got = await engine.extract([c_sec], llm, previous_sections=[p_sec])  # type: ignore[arg-type]
            count += len(got)
        return count

    llm = _ZeroLatencyLLM()
    signals = asyncio.run(diff_all(llm))
    stats = timeit(
        lambda: asyncio.run(diff_all(_ZeroLatencyLLM())),
        repeat=1 if quick else 3,
        warmup=0,
    )
    pairs = n_filings - 1
    return {
        "similarity_gate": sim_rows,
        "engine": {
            "filing_pairs": pairs,
            "avg_section_chars": round(
                sum(len(s.text) for s in history) / len(history)
            ),
            "llm_calls": llm.calls,
            "signals": signals,
            "median_s": round(stats["median_s"], 4),
            "pairs_per_s": round(pairs / stats["median_s"], 3),
        },
    }
