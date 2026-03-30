"""Signal compilation and collection utilities.

The :class:`SignalCollection` wraps a list of :class:`Signal` objects
with convenience methods for filtering, aggregation, and export.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from sigint.models import (
    Severity,
    Signal,
    SignalDirection,
    SignalType,
    SupplyChainEdge,
)

logger = structlog.get_logger()


class SignalCollection:
    """An ordered, filterable collection of extracted signals.

    This is the primary return type from :meth:`Pipeline.extract`.
    It provides a fluent API for filtering, exporting, and inspecting
    signals.

    Args:
        signals: Initial list of signals.
    """

    def __init__(self, signals: Sequence[Signal] | None = None) -> None:
        self._signals: list[Signal] = list(signals or [])

    # -- Container protocol ---------------------------------------------------

    def __len__(self) -> int:
        return len(self._signals)

    def __iter__(self) -> Iterator[Signal]:
        return iter(self._signals)

    def __getitem__(self, idx: int) -> Signal:
        return self._signals[idx]

    def __repr__(self) -> str:
        return f"SignalCollection(count={len(self._signals)})"

    # -- Mutation --------------------------------------------------------------

    def add(self, signal: Signal) -> None:
        """Append a signal to the collection."""
        self._signals.append(signal)

    def extend(self, signals: Sequence[Signal]) -> None:
        """Append multiple signals."""
        self._signals.extend(signals)

    # -- Filtering -------------------------------------------------------------

    def by_type(self, signal_type: SignalType | str) -> SignalCollection:
        """Return a new collection containing only the given signal type."""
        if isinstance(signal_type, str):
            signal_type = SignalType(signal_type)
        return SignalCollection(
            [s for s in self._signals if s.signal_type == signal_type]
        )

    def by_ticker(self, ticker: str) -> SignalCollection:
        """Return signals for a specific ticker."""
        ticker = ticker.upper()
        return SignalCollection([s for s in self._signals if s.ticker == ticker])

    def by_direction(self, direction: SignalDirection | str) -> SignalCollection:
        """Return signals with a specific direction."""
        if isinstance(direction, str):
            direction = SignalDirection(direction)
        return SignalCollection([s for s in self._signals if s.direction == direction])

    def above_strength(self, threshold: float) -> SignalCollection:
        """Return signals with strength above the threshold."""
        return SignalCollection([s for s in self._signals if s.strength >= threshold])

    def above_confidence(self, threshold: float) -> SignalCollection:
        """Return signals with confidence above the threshold."""
        return SignalCollection([s for s in self._signals if s.confidence >= threshold])

    def between(self, start: datetime, end: datetime) -> SignalCollection:
        """Return signals within a time range (inclusive)."""
        return SignalCollection(
            [s for s in self._signals if start <= s.timestamp <= end]
        )

    # -- Aggregation -----------------------------------------------------------

    def risk_changes(self, severity: str | None = None) -> SignalCollection:
        """Return risk-change signals, optionally filtered by severity.

        Args:
            severity: If given, only include changes at this severity
                level or above.
        """
        risk_signals = self.by_type(SignalType.RISK_CHANGE)
        if severity is None:
            return risk_signals

        sev = Severity(severity)
        order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
        min_idx = order.index(sev)
        allowed = set(order[min_idx:])

        return SignalCollection(
            [
                s
                for s in risk_signals
                if Severity(s.metadata.get("severity", "LOW")) in allowed
            ]
        )

    def supply_chain_edges(self) -> list[SupplyChainEdge]:
        """Extract supply-chain edges from supply_chain signals."""
        from sigint.models import FilingType, RelationType

        edges: list[SupplyChainEdge] = []
        for s in self.by_type(SignalType.SUPPLY_CHAIN):
            try:
                edges.append(
                    SupplyChainEdge(
                        source=s.ticker,
                        target=s.metadata.get("target", ""),
                        relation=RelationType(s.metadata.get("relation", "depends_on")),
                        context=s.metadata.get("edge_context", ""),
                        confidence=s.confidence,
                        filing_type=FilingType(
                            s.metadata.get("filing_type", FilingType.TEN_K.value)
                        ),
                        filed_date=s.timestamp.date(),
                    )
                )
            except (ValueError, KeyError):
                continue
        return edges

    def supply_chain_graph(self) -> Any:
        """Build a :class:`SupplyChainGraph` from supply_chain signals.

        Returns:
            A :class:`sigint.graph.SupplyChainGraph` instance.
        """
        from sigint.graph import SupplyChainGraph

        return SupplyChainGraph(self.supply_chain_edges())

    # -- Serialisation ---------------------------------------------------------

    def to_dicts(self) -> list[dict[str, Any]]:
        """Convert all signals to plain dictionaries."""
        return [s.model_dump(mode="json") for s in self._signals]

    def to_parquet(self, path: str | Path) -> Path:
        """Export signals to a Parquet file.

        Args:
            path: Destination file path.

        Returns:
            The resolved output path.
        """
        from sigint.output.parquet import write_signals_parquet

        return write_signals_parquet(self._signals, path)

    def to_csv(self, path: str | Path) -> Path:
        """Export signals to a CSV file.

        Args:
            path: Destination file path.

        Returns:
            The resolved output path.
        """
        from sigint.output.parquet import write_signals_csv

        return write_signals_csv(self._signals, path)

    def to_api(self, *, host: str = "127.0.0.1", port: int = 8080) -> None:
        """Launch a FastAPI server exposing these signals.

        Args:
            host: Bind address.
            port: Bind port.
        """
        from sigint.output.api import serve_signals

        serve_signals(self._signals, host=host, port=port)
