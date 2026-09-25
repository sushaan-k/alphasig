"""DuckDB-backed local signal storage.

Provides persistent storage for extracted signals with fast analytical
queries.  The schema mirrors :class:`Signal` and is automatically
created on first use.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

import duckdb
import pyarrow as pa
import structlog

from alphasig.exceptions import StorageError
from alphasig.models import Signal, SignalDirection, SignalType

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
            params.append(self._to_storage_timestamp(start))
        if end:
            conditions.append("timestamp <= ?")
            params.append(self._to_storage_timestamp(end))

        where = " AND ".join(conditions) if conditions else "1=1"
        sql = (
            f"SELECT timestamp, ticker, signal_type, direction, "
            f"strength, confidence, context, source_filing, "
            f"related_tickers, metadata "
            f"FROM signals WHERE {where} "
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
