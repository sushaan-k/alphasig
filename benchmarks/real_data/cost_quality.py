"""Cost and quality per filing on live data (scaffold; needs network + API key).

Usage::

    export ALPHASIG_USER_AGENT="Jane Doe jane@example.com"
    export ANTHROPIC_API_KEY=...
    python -m benchmarks.real_data.cost_quality \\
        --tickers AAPL NVDA --filings-per-ticker 2 \\
        --price-in-per-mtok <USD> --price-out-per-mtok <USD> \\
        [--labels labels.jsonl] [--out benchmarks/results/real/run.json]

Per filing it records: document size, parse time, sections found, LLM API
calls, response-cache hits, input/output/prompt-cache-read tokens (as
reported by the API), engine wall time and signals produced.  Dollar cost is computed only from the
per-million-token prices you pass -- no prices are assumed.

Quality (optional): ``--labels`` is a JSONL file of hand labels, one per
line, e.g. ``{"accession": "0000320193-24-000123", "signal_type":
"supply_chain", "match": "Taiwan Semiconductor"}``.  A label is recalled
when some signal of that type from that filing mentions ``match``
(case-insensitive) in its context or metadata; precision counts produced
signals (of labelled types, for labelled filings) that match any label.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from alphasig.edgar import EdgarClient
from alphasig.llm import DEFAULT_MODEL, LLMCache, LLMClient
from alphasig.parser import parse_filing
from alphasig.pipeline import _resolve_engines
from benchmarks._common import environment

_ENGINES = ["supply_chain", "risk_differ", "m_and_a", "tone"]


def _load_labels(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def _quality(
    rows: list[dict[str, Any]], labels: list[dict[str, Any]]
) -> dict[str, Any]:
    if not labels:
        return {"labels": 0}
    by_acc: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_acc.setdefault(r["accession"], []).extend(r["signals"])
    labelled_types = {(lab["accession"], lab["signal_type"]) for lab in labels}

    def hit(sig: dict[str, Any], needle: str) -> bool:
        hay = (sig["context"] + " " + json.dumps(sig["metadata"])).lower()
        return needle.lower() in hay

    recalled = sum(
        any(
            s["signal_type"] == lab["signal_type"] and hit(s, lab["match"])
            for s in by_acc.get(lab["accession"], [])
        )
        for lab in labels
    )
    produced = [
        (acc, s)
        for acc, sigs in by_acc.items()
        for s in sigs
        if (acc, s["signal_type"]) in labelled_types
    ]
    correct = sum(
        any(
            lab["accession"] == acc
            and lab["signal_type"] == s["signal_type"]
            and hit(s, lab["match"])
            for lab in labels
        )
        for acc, s in produced
    )
    return {
        "labels": len(labels),
        "recall": recalled / len(labels),
        "produced_in_scope": len(produced),
        "precision": (correct / len(produced)) if produced else None,
    }


def _counters(llm: LLMClient) -> dict[str, int]:
    return {
        "api_calls": llm.api_calls,
        "cache_hits": llm.cache_hits,
        "input_tokens": llm.input_tokens,
        "output_tokens": llm.output_tokens,
        "cache_read_tokens": llm.cache_read_tokens,
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    engines = _resolve_engines(args.engines)
    llm = LLMClient(
        model=args.model,
        cache=LLMCache(args.llm_cache_dir) if args.llm_cache_dir else None,
        max_concurrency=args.llm_concurrency,
    )
    rows: list[dict[str, Any]] = []
    async with (
        EdgarClient(
            os.environ["ALPHASIG_USER_AGENT"], cache_dir=args.cache_dir
        ) as edgar,
        contextlib.aclosing(llm),
    ):
        for ticker in args.tickers:
            filings = await edgar.get_filings(
                ticker, filing_types=args.filing_types, lookback_years=args.lookback
            )
            filings = filings[-(args.filings_per_ticker + 1) :]  # +1 as diff baseline
            prev_sections = None
            for idx, filing in enumerate(filings):
                filing = await edgar.fetch_filing_html(filing)
                t0 = time.perf_counter()
                sections = parse_filing(filing)
                parse_s = time.perf_counter() - t0
                if idx == 0 and len(filings) > args.filings_per_ticker:
                    prev_sections = sections
                    continue
                before = _counters(llm)
                t0 = time.perf_counter()
                results = await asyncio.gather(
                    *[
                        e.extract(sections, llm, previous_sections=prev_sections)
                        for e in engines
                    ],
                    return_exceptions=True,
                )
                engine_s = time.perf_counter() - t0
                after = _counters(llm)
                delta = {k: after[k] - before[k] for k in after}
                signals = [s for r in results if isinstance(r, list) for s in r]
                cost = None
                if (
                    args.price_in_per_mtok is not None
                    and args.price_out_per_mtok is not None
                ):
                    cost = (
                        delta["input_tokens"] * args.price_in_per_mtok
                        + delta["output_tokens"] * args.price_out_per_mtok
                    ) / 1e6
                rows.append(
                    {
                        "ticker": ticker,
                        "accession": filing.accession_number,
                        "form": filing.filing_type.value,
                        "html_mb": len(filing.raw_html.encode()) / 1e6,
                        "parse_s": parse_s,
                        "sections": [s.section_key for s in sections],
                        "engine_wall_s": engine_s,
                        "engine_errors": [
                            str(r) for r in results if isinstance(r, BaseException)
                        ],
                        **delta,
                        "cost_usd": cost,
                        "signals": [s.model_dump(mode="json") for s in signals],
                    }
                )
                prev_sections = sections
    return {"filings": rows, "quality": _quality(rows, _load_labels(args.labels))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tickers", nargs="+", required=True)
    ap.add_argument("--filing-types", nargs="+", default=["10-K"])
    ap.add_argument("--filings-per-ticker", type=int, default=2)
    ap.add_argument("--lookback", type=int, default=3)
    ap.add_argument("--engines", nargs="+", default=_ENGINES)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--llm-concurrency", type=int, default=4)
    ap.add_argument("--llm-cache-dir", default=None)
    ap.add_argument("--cache-dir", default="./edgar_cache")
    ap.add_argument("--price-in-per-mtok", type=float, default=None)
    ap.add_argument("--price-out-per-mtok", type=float, default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--out", default="benchmarks/results/real/cost_quality.json")
    args = ap.parse_args(argv)

    missing = [
        v for v in ("ALPHASIG_USER_AGENT", "ANTHROPIC_API_KEY") if not os.environ.get(v)
    ]
    if missing:
        print(
            f"refusing to run: set {', '.join(missing)} (this harness uses live EDGAR and a paid LLM)",
            file=sys.stderr,
        )
        return 2
    result = {
        "environment": environment(),
        "args": vars(args),
        **asyncio.run(_run(args)),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
