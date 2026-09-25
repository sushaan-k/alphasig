"""DuckDB-backed local signal storage.

Provides persistent storage for extracted signals with fast analytical
queries.  The schema mirrors :class:`Signal` and is automatically
created on first use.
"""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

import duckdb
import pyarrow as pa
import structlog

from alphasig.exceptions import StorageError
from alphasig.models import Signal, SignalDirection, SignalType
from alphasig.output.parquet import _SCHEMA as _ARROW_SCHEMA

logger = structlog.get_logger()

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS signals (
    id              INTEGER PRIMARY KEY DEFAULT(nextval('signal_seq')),
    timestamp       TIMESTAMP NOT NULL,
    ticker          VARCHAR NOT NULL,
    signal_type     VARCHAR NOT NULL,
    direction       VARCHAR NOT NULL,
    strength        DOUBLE NOT NULL,
    confidence      DOUBLE NOT NULL,
    context         VARCHAR,
    source_filing   VARCHAR,
    related_tickers VARCHAR,
    metadata        VARCHAR,
    inserted_at     TIMESTAMP DEFAULT current_timestamp
);
"""

_CREATE_SEQUENCE = "CREATE SEQUENCE IF NOT EXISTS signal_seq START 1;"

# One row per (filing, engine) job that finished, written in the same
# transaction as the job's signals, so incremental runs can skip work that
# is already done (see ``Pipeline.extract(incremental=True)``).
_CREATE_EXTRACTION_LOG = """\
CREATE TABLE IF NOT EXISTS extraction_log (
    accession           VARCHAR NOT NULL,
    engine              VARCHAR NOT NULL,
    ticker              VARCHAR,
    previous_accession  VARCHAR,
    calibrated          BOOLEAN NOT NULL DEFAULT FALSE,
    signal_count        INTEGER,
    completed_at        TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (accession, engine)
);
"""

_COLUMNS = (
    "timestamp, ticker, signal_type, direction, strength, "
    "confidence, context, source_filing, related_tickers, metadata"
)

# Bulk insert from a registered Arrow table, skipping signals that are
# already stored so re-running an extraction does not duplicate rows.
_INSERT_NEW = """\
INSERT INTO signals
    (timestamp, ticker, signal_type, direction, strength,
     confidence, context, source_filing, related_tickers, metadata)
SELECT DISTINCT ON (timestamp, ticker, signal_type, direction, context,
                    source_filing)
    timestamp, ticker, signal_type, direction, strength,
    confidence, context, source_filing, related_tickers, metadata
FROM _alphasig_staged AS s
WHERE NOT EXISTS (
    SELECT 1 FROM signals AS t
    WHERE t.timestamp = s.timestamp
      AND t.ticker = s.ticker
      AND t.signal_type = s.signal_type
      AND t.direction = s.direction
      AND t.context IS NOT DISTINCT FROM s.context
      AND t.source_filing IS NOT DISTINCT FROM s.source_filing
);
"""


@dataclass(frozen=True)
class ExtractionRecord:
    """A finished (filing, engine) extraction job in the ``extraction_log``.

    Attributes:
        accession: Accession number of the filing.
        engine: Engine name, e.g. ``"risk_differ"``.
        ticker: Ticker the filing belongs to.
        signal_count: Signals the job produced (after calibration).
        previous_accession: The prior filing the job compared against
            (diff engines only), ``None`` when there was none.
        calibrated: Whether the job's signals were re-scored by Jev.
    """

    accession: str
    engine: str
    ticker: str
    signal_count: int
    previous_accession: str | None = None
    calibrated: bool = False


def _where_clause(
    *,
    ticker: str | None = None,
    signal_type: str | None = None,
    direction: str | None = None,
    min_strength: float | None = None,
    min_confidence: float | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> tuple[str, list[Any]]:
    """Build the WHERE clause shared by ``query`` and ``to_arrow``."""
    conditions: list[str] = []
    params: list[Any] = []

    if ticker:
        conditions.append("ticker = ?")
        params.append(ticker.upper())
    if signal_type:
        conditions.append("signal_type = ?")
        params.append(signal_type)
    if direction:
        conditions.append("direction = ?")
        params.append(direction)
    if min_strength is not None:
        conditions.append("strength >= ?")
        params.append(min_strength)
    if min_confidence is not None:
        conditions.append("confidence >= ?")
        params.append(min_confidence)
    # Timestamps are stored as naive UTC; binding an aware datetime would
    # make DuckDB compare in the session's local time zone.
    if start:
        conditions.append("timestamp >= ?")
        params.append(SignalStore._to_storage_timestamp(start))
    if end:
        conditions.append("timestamp <= ?")
        params.append(SignalStore._to_storage_timestamp(end))

    return (" AND ".join(conditions) if conditions else "1=1"), params


class SignalStore:
    """DuckDB-backed storage for :class:`Signal` objects.

    Args:
        db_path: Path to the DuckDB database file.
            Use ``":memory:"`` for an in-memory database.
    """

    def __init__(self, db_path: str | Path = "alphasig.duckdb") -> None:
        self._db_path = str(db_path)
        try:
            self._conn = duckdb.connect(self._db_path)
            self._conn.execute(_CREATE_SEQUENCE)
            self._conn.execute(_CREATE_TABLE)
            self._conn.execute(_CREATE_EXTRACTION_LOG)
        except duckdb.Error as exc:
            raise StorageError(
                f"Failed to initialise DuckDB at {self._db_path}"
            ) from exc

    def __enter__(self) -> SignalStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def insert(self, signals: Sequence[Signal]) -> int:
        """Insert signals into the store.

        A signal already stored with the same timestamp, ticker, type,
        direction, context and source filing is skipped, so re-running an
        extraction over overlapping filings is idempotent.

        Args:
            signals: Signals to persist.

        Returns:
            Number of new signals inserted.
        """
        if not signals:
            return 0
        staged = pa.table(
            {
                "timestamp": pa.array(
                    [self._to_storage_timestamp(s.timestamp) for s in signals],
                    type=pa.timestamp("us"),
                ),
                "ticker": [s.ticker for s in signals],
                "signal_type": [s.signal_type.value for s in signals],
                "direction": [s.direction.value for s in signals],
                "strength": pa.array([s.strength for s in signals], pa.float64()),
                "confidence": pa.array([s.confidence for s in signals], pa.float64()),
                "context": [s.context for s in signals],
                "source_filing": [s.source_filing for s in signals],
                "related_tickers": [json.dumps(s.related_tickers) for s in signals],
                "metadata": [json.dumps(s.metadata) for s in signals],
            }
        )
        try:
            self._conn.register("_alphasig_staged", staged)
            try:
                row = self._conn.execute(_INSERT_NEW).fetchone()
            finally:
                self._conn.unregister("_alphasig_staged")
        except duckdb.Error as exc:
            raise StorageError(f"Failed to insert signals: {exc}") from exc
        inserted = int(row[0]) if row else 0
        logger.info("signals_stored", count=inserted, skipped=len(signals) - inserted)
        return inserted

    def query(
        self,
        *,
        ticker: str | None = None,
        signal_type: str | None = None,
        direction: str | None = None,
        min_strength: float | None = None,
        min_confidence: float | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 1000,
    ) -> list[Signal]:
        """Query stored signals with optional filters.

        Args:
            ticker: Filter by ticker symbol.
            signal_type: Filter by signal type.
            direction: Filter by direction.
            min_strength: Minimum signal strength.
            min_confidence: Minimum confidence.
            start: Earliest timestamp (inclusive).
            end: Latest timestamp (inclusive).
            limit: Maximum results.

        Returns:
            List of matching signals.
        """
        where, params = _where_clause(
            ticker=ticker,
            signal_type=signal_type,
            direction=direction,
            min_strength=min_strength,
            min_confidence=min_confidence,
            start=start,
            end=end,
        )
        sql = (
            f"SELECT {_COLUMNS} FROM signals WHERE {where} "
            f"ORDER BY timestamp DESC LIMIT ?"
        )
        params.append(limit)

        try:
            result = self._conn.execute(sql, params).fetchall()
        except duckdb.Error as exc:
            raise StorageError(f"Query failed: {exc}") from exc

        return [
            Signal(
                timestamp=self._from_storage_timestamp(row[0]),
                ticker=row[1],
                signal_type=SignalType(row[2]),
                direction=SignalDirection(row[3]),
                strength=row[4],
                confidence=row[5],
                context=row[6] or "",
                source_filing=row[7] or "",
                related_tickers=json.loads(row[8]) if row[8] else [],
                metadata=json.loads(row[9]) if row[9] else {},
            )
            for row in result
        ]

    def to_arrow(
        self,
        *,
        ticker: str | None = None,
        signal_type: str | None = None,
        direction: str | None = None,
        min_strength: float | None = None,
        min_confidence: float | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> pa.Table:
        """Return matching signals as a :class:`pyarrow.Table`, newest first.

        Takes the same filters as :meth:`query`; ``limit=None`` (the
        default) returns every match.  The schema matches the Parquet
        export (:meth:`SignalCollection.to_parquet`): ``timestamp`` is
        UTC-aware and ``related_tickers`` / ``metadata`` are JSON strings.
        No :class:`Signal` objects are built, so this is the fast path for
        analytics over large stores.
        """
        where, params = _where_clause(
            ticker=ticker,
            signal_type=signal_type,
            direction=direction,
            min_strength=min_strength,
            min_confidence=min_confidence,
            start=start,
            end=end,
        )
        sql = f"SELECT {_COLUMNS} FROM signals WHERE {where} ORDER BY timestamp DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        try:
            result = self._conn.execute(sql, params).arrow()
        except duckdb.Error as exc:
            raise StorageError(f"Query failed: {exc}") from exc
        # DuckDB >= 1.4 returns a RecordBatchReader, older versions a Table.
        table = (
            result.read_all() if isinstance(result, pa.RecordBatchReader) else result
        )
        # Stored timestamps are naive UTC; label them as such.
        return table.cast(_ARROW_SCHEMA)

    def to_pandas(self, **filters: Any) -> Any:
        """Return matching signals as a :class:`pandas.DataFrame`.

        Accepts the same keyword filters as :meth:`to_arrow`.  pandas is an
        optional dependency: ``pip install "alphasig[pandas]"``.

        Raises:
            ImportError: If pandas is not installed.
        """
        if importlib.util.find_spec("pandas") is None:
            raise ImportError(
                "SignalStore.to_pandas() needs pandas: pip install "
                "'alphasig[pandas]' (or use to_arrow(), which needs only pyarrow)."
            )
        return self.to_arrow(**filters).to_pandas()

    def completed_extractions(self) -> dict[tuple[str, str], ExtractionRecord]:
        """Return the finished extraction jobs keyed by ``(accession, engine)``."""
        try:
            rows = self._conn.execute(
                "SELECT accession, engine, ticker, signal_count, "
                "previous_accession, calibrated FROM extraction_log"
            ).fetchall()
        except duckdb.Error as exc:
            raise StorageError(f"Query failed: {exc}") from exc
        return {
            (str(row[0]), str(row[1])): ExtractionRecord(
                accession=str(row[0]),
                engine=str(row[1]),
                ticker=str(row[2] or ""),
                signal_count=int(row[3] or 0),
                previous_accession=row[4],
                calibrated=bool(row[5]),
            )
            for row in rows
        }

    def record_extraction(
        self,
        signals: Sequence[Signal],
        jobs: Sequence[ExtractionRecord],
        *,
        replace: Sequence[tuple[str, str]] = (),
    ) -> int:
        """Atomically store *signals* and mark *jobs* complete.

        Either everything is written or nothing is, so an interrupted run
        never leaves a job's signals without its log row (or the reverse).

        Args:
            signals: Signals the jobs produced.
            jobs: Every job that finished, including jobs with no signals.
                A job already in the log is overwritten.
            replace: ``(accession, signal_type)`` pairs whose previously
                stored signals are deleted first, for jobs that are being
                re-run (matched on the ``_filing_accession`` metadata the
                pipeline stamps on every signal).

        Returns:
            Number of new signals inserted.
        """
        try:
            self._conn.begin()
            try:
                if replace:
                    self._delete_job_signals(replace)
                inserted = self.insert(signals)
                if jobs:
                    self._write_log(jobs)
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
        except duckdb.Error as exc:
            raise StorageError(f"Failed to record extraction: {exc}") from exc
        return inserted

    def _delete_job_signals(self, replace: Sequence[tuple[str, str]]) -> None:
        staged = pa.table(
            {
                "accession": [acc for acc, _ in replace],
                "signal_type": [stype for _, stype in replace],
            }
        )
        self._conn.register("_alphasig_replace", staged)
        try:
            self._conn.execute(
                "DELETE FROM signals AS s WHERE EXISTS ("
                "SELECT 1 FROM _alphasig_replace AS r "
                "WHERE r.signal_type = s.signal_type AND r.accession = "
                "json_extract_string(s.metadata, '$._filing_accession'))"
            )
        finally:
            self._conn.unregister("_alphasig_replace")

    def _write_log(self, jobs: Sequence[ExtractionRecord]) -> None:
        staged = pa.table(
            {
                "accession": [j.accession for j in jobs],
                "engine": [j.engine for j in jobs],
                "ticker": [j.ticker for j in jobs],
                "previous_accession": pa.array(
                    [j.previous_accession for j in jobs], pa.string()
                ),
                "calibrated": pa.array([j.calibrated for j in jobs], pa.bool_()),
                "signal_count": pa.array([j.signal_count for j in jobs], pa.int32()),
            }
        )
        self._conn.register("_alphasig_log", staged)
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO extraction_log "
                "(accession, engine, ticker, previous_accession, calibrated, "
                "signal_count, completed_at) "
                "SELECT accession, engine, ticker, previous_accession, calibrated, "
                "signal_count, current_timestamp FROM _alphasig_log"
            )
        finally:
            self._conn.unregister("_alphasig_log")

    @staticmethod
    def _to_storage_timestamp(timestamp: datetime) -> datetime:
        """Normalize timestamps to naive UTC before storing in DuckDB."""
        if timestamp.tzinfo is None:
            return timestamp
        return timestamp.astimezone(UTC).replace(tzinfo=None)

    @staticmethod
    def _from_storage_timestamp(timestamp: datetime) -> datetime:
        """Restore UTC tzinfo after reading timestamps from DuckDB."""
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=UTC)
        return timestamp.astimezone(UTC)

    def count(self) -> int:
        """Return the total number of stored signals."""
        result = self._conn.execute("SELECT COUNT(*) FROM signals").fetchone()
        return result[0] if result else 0

    def summary(self) -> dict[str, Any]:
        """Return aggregate statistics about stored signals."""
        try:
            rows = self._conn.execute(
                "SELECT signal_type, direction, COUNT(*), "
                "AVG(strength), AVG(confidence) "
                "FROM signals GROUP BY signal_type, direction "
                "ORDER BY signal_type, direction"
            ).fetchall()
        except duckdb.Error as exc:
            raise StorageError(f"Summary query failed: {exc}") from exc

        return {
            "total": self.count(),
            "by_type_direction": [
                {
                    "signal_type": row[0],
                    "direction": row[1],
                    "count": row[2],
                    "avg_strength": round(row[3], 4),
                    "avg_confidence": round(row[4], 4),
                }
                for row in rows
            ],
        }

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
