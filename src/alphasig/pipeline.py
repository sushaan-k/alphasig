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
from alphasig.jev import JevCalibrator
from alphasig.llm import DEFAULT_MODEL, LLMCache, LLMClient
from alphasig.models import Filing, FilingSection, Signal, SignalType
from alphasig.parser import parse_filing
from alphasig.signals import SignalCollection
from alphasig.storage import ExtractionRecord, SignalStore

logger = structlog.get_logger()

_ENGINE_REGISTRY: dict[str, type[BaseEngine]] = {
    "supply_chain": SupplyChainEngine,
    "risk_differ": RiskDifferEngine,
    "m_and_a": MandAEngine,
    "tone": ToneEngine,
}

# Engines that need the previous filing for comparison
_DIFF_ENGINES = {"risk_differ", "tone"}

# The signal type each engine emits; used to replace a re-run job's output.
_ENGINE_SIGNAL_TYPE: dict[str, SignalType] = {
    "supply_chain": SignalType.SUPPLY_CHAIN,
    "risk_differ": SignalType.RISK_CHANGE,
    "m_and_a": SignalType.M_AND_A,
    "tone": SignalType.TONE_SHIFT,
}

_Job = tuple[Filing, list[FilingSection], Filing | None, list[FilingSection] | None]


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
        calibrator: Optional :class:`~alphasig.jev.JevCalibrator` that
            replaces each signal's LLM confidence with Jev's calibrated
            probability.  The caller owns it (and closes it).
        llm_concurrency: Maximum in-flight LLM requests across the whole
            run (all tickers, filings and engines).  Defaults to 8.
        llm_cache_dir: Directory for a persistent LLM response cache keyed
            by a hash of the full request (model, prompts, parameters).
            Re-running over the same filings then makes no API calls for
            work already done.  ``None`` (the default) disables it.
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
        calibrator: JevCalibrator | None = None,
        llm_concurrency: int = 8,
        llm_cache_dir: str | None = None,
    ) -> None:
        if llm_concurrency < 1:
            raise ValueError("llm_concurrency must be at least 1")
        self._model = model
        self._api_key = api_key
        self._user_agent = user_agent or os.environ.get("ALPHASIG_USER_AGENT", "")
        self._cache_dir = cache_dir
        self._db_path = db_path
        self._concurrency = concurrency
        self._max_concurrent = max_concurrent
        self._calibrator = calibrator
        self._llm_concurrency = llm_concurrency
        self._llm_cache_dir = llm_cache_dir

    async def extract(
        self,
        tickers: Sequence[str],
        *,
        filing_types: Sequence[str] | None = None,
        lookback_years: int = 3,
        engines: Sequence[str] | None = None,
        store: bool = True,
        max_concurrent: int | None = None,
        incremental: bool = False,
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
            incremental: Resume / update mode.  Every finished (filing,
                engine) job is recorded in the database's ``extraction_log``
                together with its signals, one ticker at a time, and jobs
                already recorded are skipped, so re-running after a crash or
                on a schedule only does the missing work.  A recorded job is
                redone (replacing its stored signals) when a diff engine's
                previous filing has changed, or when this run calibrates and
                Jev did not score all of the recorded job's signals.  The
                returned collection holds
                only newly extracted signals.  Requires ``db_path`` and
                ``store=True``.

        Returns:
            A :class:`SignalCollection` containing all extracted signals.

        Raises:
            ConfigurationError: If the API key or User-Agent is missing, or
                ``incremental`` is set without a database.
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

        if incremental and not (store and self._db_path):
            raise ConfigurationError(
                "incremental=True needs a database: set db_path and store=True."
            )

        if self._calibrator is not None:
            self._calibrator.connect()

        llm_cache = LLMCache(self._llm_cache_dir) if self._llm_cache_dir else None
        log_store: SignalStore | None = None
        done: dict[tuple[str, str], ExtractionRecord] | None = None
        if incremental:
            assert self._db_path is not None
            log_store = SignalStore(self._db_path)
            try:
                done = log_store.completed_extractions()
            except StorageError:
                log_store.close()
                raise

        concurrency_limit = max_concurrent or self._max_concurrent
        llm = LLMClient(
            api_key=self._api_key,
            model=self._model,
            max_concurrency=self._llm_concurrency,
            cache=llm_cache,
        )
        collection = SignalCollection()

        async with (
            _closing_store(log_store),
            _closing(llm),
            EdgarClient(
                user_agent=self._user_agent,
                cache_dir=self._cache_dir,
            ) as edgar,
        ):
            sem = asyncio.Semaphore(concurrency_limit)

            async def _process_with_limit(ticker: str) -> list[Signal]:
                async with sem:
                    signals, finished = await self._process_ticker(
                        ticker=ticker,
                        edgar=edgar,
                        llm=llm,
                        engines=active_engines,
                        filing_types=filing_types,
                        lookback_years=lookback_years,
                        done=done,
                    )
                if log_store is not None and finished:
                    _record(log_store, ticker, signals, finished, done or {})
                return signals

            logger.info(
                "pipeline_starting",
                tickers=list(tickers),
                max_concurrent=concurrency_limit,
                incremental=incremental,
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

        logger.info(
            "llm_usage",
            api_calls=llm.api_calls,
            cache_hits=llm.cache_hits,
            input_tokens=llm.input_tokens,
            output_tokens=llm.output_tokens,
        )

        if not incremental and store and self._db_path and len(collection) > 0:
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
        done: dict[tuple[str, str], ExtractionRecord] | None = None,
    ) -> tuple[list[Signal], list[ExtractionRecord]]:
        """Fetch filings, parse, and run engines for a single ticker.

        With *done* (incremental mode), (filing, engine) jobs it already
        covers are skipped and only the filings still needed are fetched.

        Returns:
            The ticker's signals, and a record of every job that finished.
        """
        logger.info("processing_ticker", ticker=ticker)

        filings = await edgar.get_filings(
            ticker,
            filing_types=filing_types,
            lookback_years=lookback_years,
        )
        if not filings:
            logger.warning("no_filings_found", ticker=ticker)
            return [], []

        if done is not None:
            filings = self._filings_to_fetch(filings, engines, done)
            if not filings:
                logger.info("ticker_up_to_date", ticker=ticker)
                return [], []

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
            return [], []

        # Group by filing type for diff engines
        by_type: dict[str, list[tuple[Filing, list[FilingSection]]]] = defaultdict(list)
        for filing, sections in parsed:
            by_type[filing.filing_type.value].append((filing, sections))

        # Pair each filing with its predecessor of the same form type.  The
        # pairs only depend on parsed sections, so every filing's engines can
        # run at once; the LLM client bounds the number of in-flight calls.
        jobs: list[tuple[_Job, list[BaseEngine]]] = []
        for filing_groups in by_type.values():
            filing_groups.sort(key=lambda x: x[0].filed_date)
            for idx, (filing, sections) in enumerate(filing_groups):
                prev_filing, prev_sections = (
                    filing_groups[idx - 1] if idx > 0 else (None, None)
                )
                pending = [
                    engine
                    for engine in engines
                    if done is None
                    or not self._is_done(done, engine.name, filing, prev_filing)
                ]
                if pending:
                    jobs.append(
                        ((filing, sections, prev_filing, prev_sections), pending)
                    )

        tasks = [
            (job_idx, engine)
            for job_idx, (_, pending) in enumerate(jobs)
            for engine in pending
        ]
        results = await asyncio.gather(
            *(
                _run_engine(
                    engine=engine,
                    sections=jobs[job_idx][0][1],
                    llm=llm,
                    previous_sections=(
                        jobs[job_idx][0][3] if engine.name in _DIFF_ENGINES else None
                    ),
                )
                for job_idx, engine in tasks
            ),
            return_exceptions=True,
        )

        per_job: list[list[Signal]] = [[] for _ in jobs]
        finished: list[list[str]] = [[] for _ in jobs]
        for (job_idx, engine), engine_result in zip(tasks, results, strict=True):
            if isinstance(engine_result, BaseException):
                logger.warning("engine_failed", ticker=ticker, error=str(engine_result))
                continue
            per_job[job_idx].extend(engine_result)
            finished[job_idx].append(engine.name)

        if self._calibrator is not None:
            per_job = await asyncio.gather(
                *(
                    self._calibrator.calibrate(signals, sections, previous)
                    for signals, ((_, sections, _, previous), _) in zip(
                        per_job, jobs, strict=True
                    )
                )
            )

        all_signals: list[Signal] = []
        records: list[ExtractionRecord] = []
        for ((filing, _, prev_filing, _), _), job_signals, engine_names in zip(
            jobs, per_job, finished, strict=True
        ):
            for sig in job_signals:
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
            for name in engine_names:
                produced = [
                    s for s in job_signals if s.signal_type is _ENGINE_SIGNAL_TYPE[name]
                ]
                records.append(
                    ExtractionRecord(
                        accession=filing.accession_number,
                        engine=name,
                        ticker=ticker,
                        signal_count=len(produced),
                        previous_accession=(
                            prev_filing.accession_number
                            if prev_filing is not None and name in _DIFF_ENGINES
                            else None
                        ),
                        # Only when Jev scored every signal: a failed Jev
                        # request keeps the LLM confidence, and a later
                        # calibrated incremental run should retry it.
                        calibrated=self._calibrator is not None
                        and all(
                            s.metadata.get("confidence_source") == "jev"
                            for s in produced
                        ),
                    )
                )

        all_signals = _deduplicate_amendment_signals(all_signals)

        logger.info(
            "ticker_complete",
            ticker=ticker,
            filings=len(parsed),
            jobs=len(tasks),
            signals=len(all_signals),
        )
        return all_signals, records

    def _is_done(
        self,
        done: dict[tuple[str, str], ExtractionRecord],
        engine: str,
        filing: Filing,
        previous: Filing | None,
    ) -> bool:
        """Whether an earlier run already covers this (filing, engine) job."""
        record = done.get((filing.accession_number, engine))
        if record is None:
            return False
        if self._calibrator is not None and not record.calibrated:
            return False
        if engine in _DIFF_ENGINES:
            previous_accession = previous.accession_number if previous else None
            return record.previous_accession == previous_accession
        return True

    def _filings_to_fetch(
        self,
        filings: list[Filing],
        engines: list[BaseEngine],
        done: dict[tuple[str, str], ExtractionRecord],
    ) -> list[Filing]:
        """Filings with unfinished jobs, plus the predecessors they diff against."""
        by_type: dict[str, list[Filing]] = defaultdict(list)
        for filing in filings:
            by_type[filing.filing_type.value].append(filing)
        needed: set[str] = set()
        for group in by_type.values():
            group.sort(key=lambda f: f.filed_date)
            for idx, filing in enumerate(group):
                previous = group[idx - 1] if idx > 0 else None
                pending = [
                    e.name
                    for e in engines
                    if not self._is_done(done, e.name, filing, previous)
                ]
                if not pending:
                    continue
                needed.add(filing.accession_number)
                if previous is not None and _DIFF_ENGINES.intersection(pending):
                    needed.add(previous.accession_number)
        return [f for f in filings if f.accession_number in needed]


def _record(
    store: SignalStore,
    ticker: str,
    signals: list[Signal],
    jobs: list[ExtractionRecord],
    done: dict[tuple[str, str], ExtractionRecord],
) -> None:
    """Persist one ticker's finished jobs and their signals atomically.

    Jobs that an earlier run had recorded are being redone: their previous
    signals are replaced rather than kept alongside the new ones.
    """
    replace = [
        (job.accession, _ENGINE_SIGNAL_TYPE[job.engine].value)
        for job in jobs
        if (job.accession, job.engine) in done
    ]
    try:
        inserted = store.record_extraction(signals, jobs, replace=replace)
    except StorageError as exc:
        # The signals are still returned; the jobs stay unrecorded and are
        # retried by the next incremental run.
        logger.error("storage_failed", ticker=ticker, error=str(exc))
        return
    logger.info(
        "extraction_recorded",
        ticker=ticker,
        jobs=len(jobs),
        replaced=len(replace),
        inserted=inserted,
    )


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


@contextlib.asynccontextmanager
async def _closing_store(
    store: SignalStore | None,
) -> AsyncIterator[SignalStore | None]:
    try:
        yield store
    finally:
        if store is not None:
            store.close()


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
