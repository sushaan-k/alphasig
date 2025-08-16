"""Main orchestration pipeline for sigint.

The :class:`Pipeline` class ties together EDGAR ingestion, section
parsing, extraction engines, and signal compilation into a single
``await pipeline.extract(...)`` call.
"""

from __future__ import annotations

import asyncio
import os
from collections import defaultdict
from collections.abc import Sequence

import structlog

from sigint.edgar import EdgarClient
from sigint.engines.base import BaseEngine
from sigint.engines.m_and_a import MandAEngine
from sigint.engines.risk_differ import RiskDifferEngine
from sigint.engines.supply_chain import SupplyChainEngine
from sigint.engines.tone import ToneEngine
from sigint.exceptions import ConfigurationError, ExtractionError, PipelineError
from sigint.llm import LLMClient
from sigint.models import Filing, FilingSection, Signal
from sigint.parser import parse_filing
from sigint.signals import SignalCollection
from sigint.storage import SignalStore

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
    """Orchestrates the full sigint extraction pipeline.

    Args:
        model: LLM model identifier.
        api_key: Anthropic API key (or read from env).
        user_agent: EDGAR User-Agent string (name + email).
        cache_dir: EDGAR cache directory.
        db_path: DuckDB storage path; ``None`` disables persistence.
        concurrency: Maximum concurrent engine tasks per filing.
        max_concurrent: Maximum number of tickers to download and
            process in parallel.  Uses :func:`asyncio.gather` with a
            semaphore so EDGAR rate limits are respected.  Defaults to 3.
    """

    def __init__(
        self,
        *,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        user_agent: str = "sigint research bot research@example.com",
        cache_dir: str = "./edgar_cache",
        db_path: str | None = "sigint.duckdb",
        concurrency: int = 4,
        max_concurrent: int = 3,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._user_agent = user_agent
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

        concurrency_limit = max_concurrent or self._max_concurrent
        llm = LLMClient(api_key=self._api_key, model=self._model)
        collection = SignalCollection()

        async with EdgarClient(
            user_agent=self._user_agent,
            cache_dir=self._cache_dir,
        ) as edgar:
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
            try:
                signal_store = SignalStore(self._db_path)
                signal_store.insert(list(collection))
                signal_store.close()
            except Exception as exc:
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
                sections = parse_filing(filing)
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

        # Run engines across filings
        all_signals: list[Signal] = []
        for _ftype_key, filing_groups in by_type.items():
            # Sort by date to get chronological order
            filing_groups.sort(key=lambda x: x[0].filed_date)

            for idx, (filing, sections) in enumerate(filing_groups):
                previous_sections = filing_groups[idx - 1][1] if idx > 0 else None

                engine_tasks = []
                for engine in engines:
                    needs_prev = engine.name in _DIFF_ENGINES
                    engine_tasks.append(
                        _run_engine(
                            engine=engine,
                            sections=sections,
                            llm=llm,
                            previous_sections=(
                                previous_sections if needs_prev else None
                            ),
                        )
                    )

                engine_results: list[
                    list[Signal] | BaseException
                ] = await asyncio.gather(*engine_tasks, return_exceptions=True)

                for engine_result in engine_results:
                    if isinstance(engine_result, BaseException):
                        logger.warning(
                            "engine_failed",
                            ticker=ticker,
                            error=str(engine_result),
                        )
                    else:
                        # Stamp each signal with the filing URL
                        for sig in engine_result:
                            stamped = sig.model_copy(
                                update={"source_filing": filing.url}
                            )
                            all_signals.append(stamped)

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
    appears in both the original filing (e.g. 10-K) and its amendment
    (10-K/A), keep only the one from the most recently filed document.

    This handles the common case where a company files a 10-K/A that
    supersedes the original 10-K -- the amended version should be the
    single source of truth.
    """
    if not signals:
        return signals

    # Key: the identity of the signal (excluding source_filing and timestamp)
    seen: dict[tuple[str, str, str, str], Signal] = {}
    for sig in signals:
        key = (sig.ticker, sig.signal_type.value, sig.direction.value, sig.context)
        existing = seen.get(key)
        if existing is None or sig.timestamp > existing.timestamp:
            seen[key] = sig

    deduped = list(seen.values())
    removed = len(signals) - len(deduped)
    if removed:
        logger.info("signals_deduplicated", removed=removed, kept=len(deduped))
    return deduped


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
