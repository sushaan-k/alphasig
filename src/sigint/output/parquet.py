"""Parquet and CSV export for signals.

Converts :class:`Signal` lists into columnar Parquet files (for use
with backtest engines like Lean or Zipline) or flat CSV.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import structlog

from sigint.models import Signal

logger = structlog.get_logger()

_SCHEMA = pa.schema(
    [
        pa.field("timestamp", pa.timestamp("us", tz="UTC")),
        pa.field("ticker", pa.string()),
        pa.field("signal_type", pa.string()),
        pa.field("direction", pa.string()),
        pa.field("strength", pa.float64()),
        pa.field("confidence", pa.float64()),
        pa.field("context", pa.string()),
        pa.field("source_filing", pa.string()),
        pa.field("related_tickers", pa.string()),
        pa.field("metadata", pa.string()),
    ]
)


def _signals_to_table(signals: Sequence[Signal]) -> pa.Table:
    """Convert signals to a PyArrow table."""
    timestamps = []
    tickers = []
    signal_types = []
    directions = []
    strengths = []
    confidences = []
    contexts = []
    source_filings = []
    related_tickers_col = []
    metadata_col = []

    for s in signals:
        timestamps.append(s.timestamp)
        tickers.append(s.ticker)
        signal_types.append(s.signal_type.value)
        directions.append(s.direction.value)
        strengths.append(s.strength)
        confidences.append(s.confidence)
        contexts.append(s.context)
        source_filings.append(s.source_filing)
        related_tickers_col.append(json.dumps(s.related_tickers))
        metadata_col.append(json.dumps(s.metadata))

    return pa.table(
        {
            "timestamp": pa.array(timestamps, type=pa.timestamp("us", tz="UTC")),
            "ticker": tickers,
            "signal_type": signal_types,
            "direction": directions,
            "strength": strengths,
            "confidence": confidences,
            "context": contexts,
            "source_filing": source_filings,
            "related_tickers": related_tickers_col,
            "metadata": metadata_col,
        },
        schema=_SCHEMA,
    )


def write_signals_parquet(signals: Sequence[Signal], path: str | Path) -> Path:
    """Write signals to a Parquet file.

    Args:
        signals: Signals to write.
        path: Output file path.

    Returns:
        Resolved output path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    table = _signals_to_table(signals)
    pq.write_table(table, str(out), compression="snappy")
    logger.info("parquet_written", path=str(out), rows=len(signals))
    return out


def write_signals_csv(signals: Sequence[Signal], path: str | Path) -> Path:
    """Write signals to a CSV file.

    Args:
        signals: Signals to write.
        path: Output file path.

    Returns:
        Resolved output path.
    """
    import pyarrow.csv as pcsv

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    table = _signals_to_table(signals)
    pcsv.write_csv(table, str(out))
    logger.info("csv_written", path=str(out), rows=len(signals))
    return out


def read_signals_parquet(path: str | Path) -> list[Signal]:
    """Read signals from a Parquet file.

    Args:
        path: Path to the Parquet file.

    Returns:
        List of :class:`Signal` instances.
    """
    from sigint.models import SignalDirection, SignalType

    table = pq.read_table(str(path))
    signals: list[Signal] = []
    for batch in table.to_batches():
        for row in batch.to_pylist():
            signals.append(
                Signal(
                    timestamp=row["timestamp"],
                    ticker=row["ticker"],
                    signal_type=SignalType(row["signal_type"]),
                    direction=SignalDirection(row["direction"]),
                    strength=row["strength"],
                    confidence=row["confidence"],
                    context=row["context"],
                    source_filing=row["source_filing"],
                    related_tickers=json.loads(row.get("related_tickers", "[]")),
                    metadata=json.loads(row.get("metadata", "{}")),
                )
            )
    return signals
