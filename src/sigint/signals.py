"""Signal compilation and collection utilities.

The :class:`SignalCollection` wraps a list of :class:`Signal` objects
with convenience methods for filtering, aggregation, and export.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
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
from sigint.sectors import Sector, classify_sector

logger = structlog.get_logger()


@dataclass(frozen=True)
class CorrelationMatrix:
    """Result of a pairwise signal-type correlation analysis.

    Attributes:
        signal_types: Ordered list of signal type labels (row / column headers).
        matrix: Square list-of-lists of Pearson correlation coefficients.
        ticker_count: Number of tickers that contributed to the calculation.
    """

    signal_types: list[str]
    matrix: list[list[float]]
    ticker_count: int = 0

    def get(self, type_a: str, type_b: str) -> float:
        """Look up correlation between two signal types."""
        idx_a = self.signal_types.index(type_a)
        idx_b = self.signal_types.index(type_b)
        return self.matrix[idx_a][idx_b]


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Compute Pearson correlation coefficient between two equal-length series.

    Returns 0.0 when either series has zero variance (avoids division by
    zero).
    """
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denom = math.sqrt(var_x * var_y)
    if denom == 0.0:
        return 0.0
    return cov / denom


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

    def by_sector(self, sector: Sector | str) -> SignalCollection:
        """Return signals whose ticker belongs to *sector*.

        Args:
            sector: A :class:`Sector` member or its string value.

        Returns:
            Filtered :class:`SignalCollection`.
        """
        if isinstance(sector, str):
            sector = Sector(sector)
        return SignalCollection(
            [s for s in self._signals if classify_sector(s.ticker) == sector]
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

    # -- Correlation -----------------------------------------------------------

    def correlate(self, *signal_types: SignalType | str) -> CorrelationMatrix:
        """Compute pairwise Pearson correlation of signal strengths across tickers.

        For each ticker, the average strength per signal type is computed.
        Then the Pearson correlation coefficient is calculated pairwise
        across all tickers that have at least one signal of each requested
        type.

        Args:
            *signal_types: Two or more signal types to compare.  Accepts
                :class:`SignalType` members or plain strings.

        Returns:
            A :class:`CorrelationMatrix` with one row/column per signal type.

        Raises:
            ValueError: If fewer than two signal types are provided.
        """
        types = [
            SignalType(t) if isinstance(t, str) else t for t in signal_types
        ]
        if len(types) < 2:
            raise ValueError("correlate() requires at least two signal types")

        # Build ticker -> signal_type -> [strength, ...]
        ticker_map: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for sig in self._signals:
            if sig.signal_type in types:
                ticker_map[sig.ticker][sig.signal_type.value].append(sig.strength)

        # Reduce to average strength per (ticker, type)
        type_labels = [t.value for t in types]

        # Only keep tickers that have data for ALL requested types
        valid_tickers = [
            tk
            for tk, type_strengths in ticker_map.items()
            if all(tl in type_strengths for tl in type_labels)
        ]

        n = len(type_labels)
        matrix: list[list[float]] = [[0.0] * n for _ in range(n)]

        if len(valid_tickers) < 2:
            # Not enough data -- return identity matrix
            for i in range(n):
                matrix[i][i] = 1.0
            return CorrelationMatrix(
                signal_types=type_labels,
                matrix=matrix,
                ticker_count=len(valid_tickers),
            )

        # Build vectors: one value per ticker for each type
        vectors: dict[str, list[float]] = {}
        for tl in type_labels:
            vectors[tl] = [
                sum(ticker_map[tk][tl]) / len(ticker_map[tk][tl])
                for tk in valid_tickers
            ]

        for i, ti in enumerate(type_labels):
            for j, tj in enumerate(type_labels):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    matrix[i][j] = _pearson(vectors[ti], vectors[tj])

        return CorrelationMatrix(
            signal_types=type_labels,
            matrix=matrix,
            ticker_count=len(valid_tickers),
        )

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
