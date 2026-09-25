"""End-to-end pipeline throughput with mocked EDGAR and a latency-injected mock LLM.

Nothing leaves the process:

* EDGAR (``company_tickers.json``, ``submissions/CIK*.json``, Archives
  documents) is served by ``respx`` with a fixed per-request network delay.
  The real ``EdgarClient`` rate limiter stays active, so the 10 req/s SEC
  cap is part of the measurement.
* ``anthropic.AsyncAnthropic`` is replaced by a fake whose
  ``messages.create`` sleeps for a fixed latency and returns deterministic,
  schema-valid JSON for each engine prompt.  The fake counts calls and the
  peak number of in-flight requests.  (Real prompt caching, retries and
  token costs are not modelled.)

Filings are synthetic (see :mod:`benchmarks.fixtures`); each ticker's
risk factors evolve between filings so the risk-differ engine calls the LLM.
Scenarios that need options the installed ``alphasig`` does not have are
skipped and reported as such, so the same file can measure old and new code.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import random
import tempfile
import time
from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import anthropic
import httpx
import respx

from alphasig.pipeline import Pipeline
from benchmarks._common import SEED
from benchmarks.fixtures import evolve_risks, risk_paragraphs, synth_filing

_UA = "alphasig-bench bench@example.com"
_ALL_ENGINES = ["supply_chain", "risk_differ", "m_and_a", "tone"]


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------


class LLMStats:
    def __init__(self) -> None:
        self.calls = 0
        self.inflight = 0
        self.max_inflight = 0
        self.input_chars = 0


def _response_for(system: str) -> list[dict[str, Any]]:
    if "supply-chain" in system:
        return [
            {
                "source": "X",
                "target": "TSM",
                "relation": "depends_on",
                "context": "wafers",
                "confidence": 0.9,
            }
        ]
    if "securities lawyer" in system:
        return [
            {
                "change_type": "ESCALATED",
                "risk": "Supply concentration",
                "language_shift": "'may' -> 'is currently'",
                "severity_estimate": "HIGH",
                "confidence": 0.8,
                "related_tickers": ["TSM"],
            }
        ]
    if "M&A" in system:
        return [
            {
                "indicator": "strategic alternatives language",
                "category": "strategic_alternatives",
                "excerpt": "exploring strategic alternatives",
                "confidence": 0.6,
            }
        ]
    if "management" in system.lower():
        return [
            {
                "topic": "revenue growth",
                "tone": "hedging_cautious",
                "confidence": 0.7,
                "key_phrases": ["may"],
            },
            {
                "topic": "margins",
                "tone": "confident_expanding",
                "confidence": 0.8,
                "key_phrases": ["strong"],
            },
        ]
    return []


def _system_text(system: Any) -> str:
    """The system prompt as text (a plain string, or a list of text blocks)."""
    if isinstance(system, str):
        return system
    return "".join(block.get("text", "") for block in system)


def fake_anthropic_factory(stats: LLMStats, latency_s: float) -> type:
    class _Messages:
        async def create(self, **kwargs: Any) -> Any:
            stats.calls += 1
            stats.inflight += 1
            stats.max_inflight = max(stats.max_inflight, stats.inflight)
            stats.input_chars += len(_system_text(kwargs.get("system", ""))) + sum(
                len(m["content"]) for m in kwargs.get("messages", [])
            )
            try:
                await asyncio.sleep(latency_s)
            finally:
                stats.inflight -= 1
            text = json.dumps(_response_for(_system_text(kwargs.get("system", ""))))
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text=text)],
                stop_reason="end_turn",
                usage=SimpleNamespace(
                    input_tokens=0, output_tokens=0, cache_read_input_tokens=0
                ),
            )

    class FakeAsyncAnthropic:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.messages = _Messages()

        async def close(self) -> None:
            return None

    return FakeAsyncAnthropic


# ---------------------------------------------------------------------------
# Mock EDGAR
# ---------------------------------------------------------------------------


class EdgarStats:
    def __init__(self) -> None:
        self.requests = 0
        self.archive_requests = 0


def build_corpus(n_tickers: int, n_filings: int, seed: int = SEED) -> dict[str, Any]:
    """Synthetic EDGAR universe: tickers -> submissions + documents."""
    rng = random.Random(seed)
    today = date.today()
    tickers = {}
    for t in range(n_tickers):
        ticker = f"BN{t:02d}"
        cik = 900000 + t
        risks_k = risk_paragraphs(rng, 24)
        risks_q = risk_paragraphs(rng, 6)
        forms, accs, fdates, rdates, docs, accepted = [], [], [], [], [], []
        documents = {}
        for i in range(n_filings):
            form = "10-K" if i % 2 == 0 else "10-Q"
            filed = today - timedelta(days=45 + 91 * (n_filings - i))
            acc = f"{cik:010d}-{filed.year % 100:02d}-{i:06d}"
            doc = f"bn{t}-{filed.isoformat()}.htm"
            if form == "10-K":
                risks_k = evolve_risks(rng, risks_k)
                risks = risks_k
            else:
                risks_q = evolve_risks(rng, risks_q)
                risks = risks_q
            html = synth_filing(
                seed * 1000 + t * 100 + i,
                form=form,
                company=f"Bench {t} Inc.",
                body_paragraphs=6,
                tables=2,
                table_rows=30,
                risks=risks,
            ).html
            forms.append(form)
            accs.append(acc)
            fdates.append(filed.isoformat())
            rdates.append((filed - timedelta(days=40)).isoformat())
            docs.append(doc)
            accepted.append(f"{filed.isoformat()}T16:{i % 60:02d}:07.000Z")
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc}"
            documents[url] = html
        tickers[ticker] = {
            "cik": cik,
            "submissions": {
                "cik": str(cik),
                "name": f"Bench {t} Inc.",
                "filings": {
                    "recent": {
                        "accessionNumber": accs,
                        "form": forms,
                        "filingDate": fdates,
                        "reportDate": rdates,
                        "acceptanceDateTime": accepted,
                        "primaryDocument": docs,
                    }
                },
            },
            "documents": documents,
        }
    return tickers


@contextlib.contextmanager
def mocked_edgar(
    corpus: dict[str, Any], stats: EdgarStats, latency_s: float
) -> Iterator[None]:
    company_tickers = {
        str(i): {"cik_str": v["cik"], "ticker": k, "title": v["submissions"]["name"]}
        for i, (k, v) in enumerate(corpus.items())
    }
    subs = {f"CIK{v['cik']:010d}.json": v["submissions"] for v in corpus.values()}
    docs: dict[str, str] = {}
    for v in corpus.values():
        docs.update(v["documents"])

    async def handler(request: httpx.Request) -> httpx.Response:
        stats.requests += 1
        await asyncio.sleep(latency_s)
        url = str(request.url)
        if url.endswith("company_tickers.json"):
            return httpx.Response(200, json=company_tickers)
        if "/submissions/" in url:
            return httpx.Response(200, json=subs[url.rsplit("/", 1)[1]])
        if url in docs:
            stats.archive_requests += 1
            return httpx.Response(200, text=docs[url])
        return httpx.Response(404)

    with respx.mock(assert_all_called=False) as router:
        router.route(host__in=["www.sec.gov", "data.sec.gov", "efts.sec.gov"]).mock(
            side_effect=handler
        )
        yield


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def _supported(cls_or_fn: Any) -> set[str]:
    return set(inspect.signature(cls_or_fn).parameters)


def _run_once(
    corpus: dict[str, Any],
    *,
    llm_latency: float,
    net_latency: float,
    pipeline_kwargs: dict[str, Any],
    extract_kwargs: dict[str, Any],
) -> dict[str, Any]:
    llm_stats, edgar_stats = LLMStats(), EdgarStats()
    fake = fake_anthropic_factory(llm_stats, llm_latency)
    with (
        mocked_edgar(corpus, edgar_stats, net_latency),
        patch.object(anthropic, "AsyncAnthropic", fake),
    ):
        pipeline = Pipeline(**pipeline_kwargs)
        t0 = time.perf_counter()
        coll = asyncio.run(
            pipeline.extract(
                tickers=list(corpus),
                filing_types=["10-K", "10-Q"],
                lookback_years=5,
                engines=_ALL_ENGINES,
                **extract_kwargs,
            )
        )
        wall = time.perf_counter() - t0
    return {
        "wall_s": round(wall, 3),
        "llm_calls": llm_stats.calls,
        "llm_max_inflight": llm_stats.max_inflight,
        "llm_input_mchars": round(llm_stats.input_chars / 1e6, 3),
        "edgar_requests": edgar_stats.requests,
        "edgar_archive_requests": edgar_stats.archive_requests,
        "signals": len(coll),
    }


def _config(quick: bool) -> tuple[int, int, float, float]:
    """(tickers, filings per ticker, LLM latency s, network latency s)."""
    return (3, 4, 0.05, 0.02) if quick else (6, 6, 0.25, 0.02)


def _base_kwargs(tmp: Path, **extra: Any) -> dict[str, Any]:
    """Pipeline kwargs, filtered to what the installed version accepts."""
    kw: dict[str, Any] = {
        "api_key": "bench-key",
        "user_agent": _UA,
        "cache_dir": str(tmp / "edgar"),
        "db_path": None,
    }
    kw.update(extra)
    pipe_params = _supported(Pipeline)
    return {k: v for k, v in kw.items() if k in pipe_params}


def run_scenario_cold_default(quick: bool = False) -> dict[str, Any]:
    """Just the ``cold_default`` scenario (used by benchmarks.pipeline_ab)."""
    n_tickers, n_filings, llm_latency, net_latency = _config(quick)
    corpus = build_corpus(n_tickers, n_filings)
    with tempfile.TemporaryDirectory() as d:
        return _run_once(
            corpus,
            llm_latency=llm_latency,
            net_latency=net_latency,
            pipeline_kwargs=_base_kwargs(Path(d)),
            extract_kwargs={"store": False},
        )


def run(quick: bool = False) -> dict[str, Any]:
    n_tickers, n_filings, llm_latency, net_latency = _config(quick)
    corpus = build_corpus(n_tickers, n_filings)
    pipe_params = _supported(Pipeline)
    extract_params = _supported(Pipeline.extract)
    base_kwargs = _base_kwargs

    out: dict[str, Any] = {
        "tickers": n_tickers,
        "filings_per_ticker": n_filings,
        "llm_latency_s": llm_latency,
        "net_latency_s": net_latency,
        "scenarios": {},
    }
    sc = out["scenarios"]

    def go(
        name: str, tmp: Path, pkw: dict[str, Any], ekw: dict[str, Any] | None = None
    ) -> None:
        ekw = {"store": False, **(ekw or {})}
        missing = [k for k in ekw if k not in extract_params]
        if missing:
            sc[name] = {"skipped": f"unsupported option(s): {missing}"}
            return
        sc[name] = _run_once(
            corpus,
            llm_latency=llm_latency,
            net_latency=net_latency,
            pipeline_kwargs=pkw,
            extract_kwargs=ekw,
        )

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        # 1. Defaults, cold EDGAR cache.
        go("cold_default", tmp, base_kwargs(tmp))
        # 2. Same again: EDGAR documents now come from the on-disk cache.
        go("warm_edgar_cache", tmp, base_kwargs(tmp))

    if "llm_concurrency" in pipe_params:
        for c in (1, 4, 16):
            with tempfile.TemporaryDirectory() as d:
                tmp = Path(d)
                go(
                    f"cold_llm_concurrency_{c}",
                    tmp,
                    base_kwargs(tmp, llm_concurrency=c),
                )
    else:
        sc["cold_llm_concurrency_*"] = {
            "skipped": "Pipeline has no llm_concurrency option"
        }

    if "llm_cache_dir" in pipe_params:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            kw = base_kwargs(tmp, llm_cache_dir=str(tmp / "llm"))
            go("llm_disk_cache_first_run", tmp, kw)
            go("llm_disk_cache_rerun", tmp, kw)
    else:
        sc["llm_disk_cache_*"] = {"skipped": "Pipeline has no llm_cache_dir option"}

    if "incremental" in extract_params:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            kw = base_kwargs(tmp, db_path=str(tmp / "signals.duckdb"))
            go("incremental_first_run", tmp, kw, {"store": True, "incremental": True})
            go("incremental_rerun", tmp, kw, {"store": True, "incremental": True})
    else:
        sc["incremental_*"] = {"skipped": "Pipeline.extract has no incremental option"}
    return out
