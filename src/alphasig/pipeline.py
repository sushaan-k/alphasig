"""Main orchestration pipeline for alphasig.

The :class:`Pipeline` class ties together EDGAR ingestion, section
parsing, extraction engines, and signal compilation into a single
``await pipeline.extract(...)`` call.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
from collections import defaultdict
from collections.abc import AsyncIterator, Sequence

import structlog

from alphasig.edgar import EdgarClient
from alphasig.engines.base import BaseEngine
from alphasig.engines.m_and_a import MandAEngine
from alphasig.engines.risk_differ import RiskDifferEngine
from alphasig.engines.supply_chain import SupplyChainEngine
from alphasig.engines.tone import ToneEngine
from alphasig.exceptions import (
    ConfigurationError,
    ExtractionError,
    PipelineError,
    StorageError,
)
from alphasig.llm import DEFAULT_MODEL, LLMClient
from alphasig.models import Filing, FilingSection, Signal
from alphasig.parser import parse_filing
from alphasig.signals import SignalCollection
from alphasig.storage import SignalStore

logger = structlog.get_logger()

_ENGINE_REGISTRY: dict[str, type[BaseEngine]] = {
    "supply_chain": SupplyChainEngine,
    "risk_differ": RiskDifferEngine,
    "m_and_a": MandAEngine,
    "tone": ToneEngine,
}

# Engines that need the previous filing for comparison
_DIFF_ENGINES = {"risk_differ", "tone"}


class Pipeline:
    """Orchestrates the full alphasig extraction pipeline.

    Args:
        model: LLM model identifier (default :data:`alphasig.llm.DEFAULT_MODEL`).
        api_key: Anthropic API key (or read from env).
        user_agent: EDGAR User-Agent identifying you, ``"Name email@domain"``
            (SEC fair-access policy).  Falls back to the
            ``ALPHASIG_USER_AGENT`` environment variable.
        cache_dir: EDGAR cache directory; ``None`` disables caching.
        db_path: DuckDB storage path; ``None`` disables persistence.
        concurrency: Maximum concurrent filing downloads per ticker.  All
            requests share one client limited to 10 requests/second.
        max_concurrent: Maximum number of tickers to process in parallel.
            Defaults to 3.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        user_agent: str | None = None,
        cache_dir: str | None = "./edgar_cache",
        db_path: str | None = "alphasig.duckdb",
        concurrency: int = 4,
        max_concurrent: int = 3,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._user_agent = user_agent or os.environ.get("ALPHASIG_USER_AGENT", "")
        self._cache_dir = cache_dir
        self._db_path = db_path
        self._concurrency = concurrency
        self._max_concurrent = max_concurrent

    async def extract(
        self,
        tickers: Sequence[str],
        *,
        filing_types: Sequence[str] | None = None,
        lookback_years: int = 3,
        engines: Sequence[str] | None = None,
        store: bool = True,
        max_concurrent: int | None = None,
    ) -> SignalCollection:
        """Run the full extraction pipeline.

        Args:
            tickers: Company ticker symbols to analyse.
            filing_types: SEC filing types to fetch (default: 10-K, 10-Q).
            lookback_years: Years of filings to retrieve.
            engines: Extraction engines to run.  Defaults to all.
            store: Whether to persist signals to DuckDB.
            max_concurrent: Maximum number of tickers to process in
                parallel.  Defaults to the value set at construction time.
                The semaphore respects EDGAR rate limits while allowing
                concurrent filing downloads via :func:`asyncio.gather`.

        Returns:
            A :class:`SignalCollection` containing all extracted signals.

        Raises:
            PipelineError: On orchestration failures.
        """
        engine_names = list(engines or _ENGINE_REGISTRY.keys())
        active_engines = _resolve_engines(engine_names)

        if not self._api_key and not os.environ.get("ANTHROPIC_API_KEY"):
            raise ConfigurationError(
                "ANTHROPIC_API_KEY environment variable is required "
                "for signal extraction."
            )
        if "@" not in self._user_agent:
            raise ConfigurationError(
                "SEC EDGAR requires a User-Agent identifying you, e.g. "
                "'Jane Doe jane@example.com'. Pass user_agent= or set "
                "ALPHASIG_USER_AGENT."
            )

        concurrency_limit = max_concurrent or self._max_concurrent
        llm = LLMClient(api_key=self._api_key, model=self._model)
        collection = SignalCollection()

        async with (
            EdgarClient(
                user_agent=self._user_agent,
                cache_dir=self._cache_dir,
            ) as edgar,
            _closing(llm),
        ):
            sem = asyncio.Semaphore(concurrency_limit)

            async def _process_with_limit(ticker: str) -> list[Signal]:
                async with sem:
                    return await self._process_ticker(
                        ticker=ticker,
                        edgar=edgar,
                        llm=llm,
                        engines=active_engines,
                        filing_types=filing_types,
                        lookback_years=lookback_years,
                    )

            logger.info(
                "pipeline_starting",
                tickers=list(tickers),
                max_concurrent=concurrency_limit,
            )

            results: list[list[Signal] | BaseException] = await asyncio.gather(
                *[_process_with_limit(t) for t in tickers],
                return_exceptions=True,
            )

            for ticker, result in zip(tickers, results, strict=True):
                if isinstance(result, BaseException):
                    logger.error(
                        "pipeline_ticker_failed",
                        ticker=ticker,
                        error=str(result),
                    )
                else:
                    collection.extend(result)

        if store and self._db_path and len(collection) > 0:
            # Extraction already spent the LLM budget: log a storage failure
            # and still return the signals rather than discarding them.
            try:
                with SignalStore(self._db_path) as signal_store:
                    signal_store.insert(list(collection))
            except StorageError as exc:
                logger.error("storage_failed", error=str(exc))

        logger.info(
            "pipeline_complete",
            tickers=list(tickers),
            total_signals=len(collection),
        )
        return collection

    async def _process_ticker(
        self,
        *,
        ticker: str,
        edgar: EdgarClient,
        llm: LLMClient,
        engines: list[BaseEngine],
        filing_types: Sequence[str] | None,
        lookback_years: int,
    ) -> list[Signal]:
        """Fetch filings, parse, and run engines for a single ticker."""
        logger.info("processing_ticker", ticker=ticker)

        filings = await edgar.get_filings(
            ticker,
            filing_types=filing_types,
            lookback_years=lookback_years,
        )
        if not filings:
            logger.warning("no_filings_found", ticker=ticker)
            return []

        # Download HTML for all filings (with concurrency limit)
        sem = asyncio.Semaphore(self._concurrency)

        async def _fetch(f: Filing) -> Filing:
            async with sem:
                return await edgar.fetch_filing_html(f)

        filings_with_html: list[Filing | BaseException] = await asyncio.gather(
            *[_fetch(f) for f in filings],
            return_exceptions=True,
        )

        # Parse all filings into sections
        parsed: list[tuple[Filing, list[FilingSection]]] = []
        for fetch_result in filings_with_html:
            if isinstance(fetch_result, BaseException):
                logger.warning("filing_fetch_failed", error=str(fetch_result))
                continue
            filing: Filing = fetch_result
            try:
                # HTML parsing is CPU-bound; keep the event loop free for the
                # other tickers' downloads and LLM calls.
                sections = await asyncio.to_thread(parse_filing, filing)
                parsed.append((filing, sections))
            except Exception as exc:
                logger.warning(
                    "filing_parse_failed",
                    accession=filing.accession_number,
                    error=str(exc),
                )

        if not parsed:
            return []

        # Group by filing type for diff engines
        by_type: dict[str, list[tuple[Filing, list[FilingSection]]]] = defaultdict(list)
        for filing, sections in parsed:
            by_type[filing.filing_type.value].append((filing, sections))

        # Pair each filing with its predecessor of the same form type.  The
        # pairs only depend on parsed sections, so every filing's engines can
        # run at once; the LLM client bounds the number of in-flight calls.
        jobs: list[tuple[Filing, list[FilingSection], list[FilingSection] | None]] = []
        for filing_groups in by_type.values():
            filing_groups.sort(key=lambda x: x[0].filed_date)
            for idx, (filing, sections) in enumerate(filing_groups):
                previous = filing_groups[idx - 1][1] if idx > 0 else None
                jobs.append((filing, sections, previous))

        results = await asyncio.gather(
            *(
                _run_engine(
                    engine=engine,
                    sections=sections,
                    llm=llm,
                    previous_sections=(
                        previous if engine.name in _DIFF_ENGINES else None
                    ),
                )
                for _, sections, previous in jobs
                for engine in engines
            ),
            return_exceptions=True,
        )

        all_signals: list[Signal] = []
        for job_idx, engine_result in enumerate(results):
            if isinstance(engine_result, BaseException):
                logger.warning("engine_failed", ticker=ticker, error=str(engine_result))
                continue
            filing = jobs[job_idx // len(engines)][0]
            for sig in engine_result:
                stamped_metadata = {
                    **sig.metadata,
                    "_filing_accession": filing.accession_number,
                    "_filing_type": filing.filing_type.value,
                    "_period_of_report": filing.period_of_report.isoformat(),
                }
                all_signals.append(
                    sig.model_copy(
                        update={
                            "source_filing": filing.url,
                            "metadata": stamped_metadata,
                        }
                    )
                )

        all_signals = _deduplicate_amendment_signals(all_signals)

        logger.info(
            "ticker_complete",
            ticker=ticker,
            filings=len(parsed),
            signals=len(all_signals),
        )
        return all_signals


def _deduplicate_amendment_signals(signals: list[Signal]) -> list[Signal]:
    """Remove duplicate signals caused by filing amendments.

    When the same signal (same ticker, signal_type, direction, and context)
    appears in both an original filing (e.g. 10-K) and its amendment
    (10-K/A), keep only the earliest one: that is when the information
    became public, so keeping the later copy would date the signal after
    the market could already act on it.
    """
    if not signals:
        return signals

    # Only dedupe true amendment families; repeated signals across separate
    # quarterly/annual filings must remain distinct for time-series analysis.
    seen: dict[tuple[str, str, str, str, str], Signal] = {}
    for sig in signals:
        key = (
            sig.ticker,
            sig.signal_type.value,
            sig.direction.value,
            sig.context,
            _filing_family(sig),
        )
        existing = seen.get(key)
        if existing is None or sig.timestamp < existing.timestamp:
            seen[key] = sig

    deduped = list(seen.values())
    removed = len(signals) - len(deduped)
    if removed:
        logger.info("signals_deduplicated", removed=removed, kept=len(deduped))
    return deduped


def _filing_family(sig: Signal) -> str:
    """Return a stable filing-family identifier for amendment dedupe.

    The family must collapse an original filing and its amendment to the
    same value while keeping unrelated recurring filings separate.
    """
    filing_type = str(sig.metadata.get("_filing_type", "")).strip().upper()
    period = str(sig.metadata.get("_period_of_report", "")).strip()
    if filing_type and period:
        return f"{_base_filing_type(filing_type)}|{period}"

    accession = str(sig.metadata.get("_filing_accession", "")).strip()
    if accession:
        return accession

    source = sig.source_filing.strip().lower()
    if not source:
        return ""

    # Normalize obvious amendment suffixes in simple URLs/labels such as
    # ``10-K-A`` or ``10-Q/A`` while preserving accession/path uniqueness.
    normalized = re.sub(r"(?i)([-_/]?a)(?=(?:$|[?#.]))", "", source)
    normalized = re.sub(r"(?i)(/a)(?=(?:$|[?#]))", "", normalized)
    return normalized


def _base_filing_type(filing_type: str) -> str:
    """Normalize amendment suffixes to the base SEC form type."""
    if filing_type.endswith("/A"):
        return filing_type[:-2]
    if filing_type.endswith("-A"):
        return filing_type[:-2]
    return filing_type


@contextlib.asynccontextmanager
async def _closing(llm: LLMClient) -> AsyncIterator[LLMClient]:
    try:
        yield llm
    finally:
        await llm.aclose()


async def _run_engine(
    *,
    engine: BaseEngine,
    sections: Sequence[FilingSection],
    llm: LLMClient,
    previous_sections: Sequence[FilingSection] | None,
) -> list[Signal]:
    """Run a single engine with error wrapping."""
    try:
        return await engine.extract(sections, llm, previous_sections=previous_sections)
    except Exception as exc:
        raise ExtractionError(f"Engine '{engine.name}' failed: {exc}") from exc


def _resolve_engines(names: Sequence[str]) -> list[BaseEngine]:
    """Instantiate engines by name."""
    engines: list[BaseEngine] = []
    for name in names:
        cls = _ENGINE_REGISTRY.get(name)
        if cls is None:
            raise PipelineError(
                f"Unknown engine: {name!r}. Available: {sorted(_ENGINE_REGISTRY)}"
            )
        engines.append(cls())
    return engines
