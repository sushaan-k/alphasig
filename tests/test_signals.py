"""Tests for sigint.signals -- SignalCollection."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from sigint.models import FilingType, Signal, SignalDirection, SignalType
from sigint.signals import CorrelationMatrix, SignalCollection, _pearson


class TestSignalCollection:
    """Tests for the SignalCollection class."""

    def test_length(self, sample_signals: list[Signal]) -> None:
        coll = SignalCollection(sample_signals)
        assert len(coll) == 4

    def test_iteration(self, sample_signals: list[Signal]) -> None:
        coll = SignalCollection(sample_signals)
        items = list(coll)
        assert len(items) == 4

    def test_indexing(self, sample_signals: list[Signal]) -> None:
        coll = SignalCollection(sample_signals)
        assert coll[0].ticker == "AAPL"

    def test_add(self) -> None:
        coll = SignalCollection()
        sig = Signal(
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            ticker="TEST",
            signal_type=SignalType.SUPPLY_CHAIN,
            direction=SignalDirection.NEUTRAL,
            strength=0.5,
            confidence=0.5,
            context="test",
            source_filing="",
        )
        coll.add(sig)
        assert len(coll) == 1

    def test_extend(self, sample_signals: list[Signal]) -> None:
        coll = SignalCollection()
        coll.extend(sample_signals)
        assert len(coll) == 4

    def test_by_type(self, sample_signals: list[Signal]) -> None:
        coll = SignalCollection(sample_signals)
        sc = coll.by_type(SignalType.SUPPLY_CHAIN)
        assert len(sc) == 1
        assert sc[0].signal_type == SignalType.SUPPLY_CHAIN

    def test_by_type_string(self, sample_signals: list[Signal]) -> None:
        coll = SignalCollection(sample_signals)
        sc = coll.by_type("supply_chain")
        assert len(sc) == 1

    def test_by_ticker(self, sample_signals: list[Signal]) -> None:
        coll = SignalCollection(sample_signals)
        aapl = coll.by_ticker("AAPL")
        assert len(aapl) == 2
        assert all(s.ticker == "AAPL" for s in aapl)

    def test_by_direction(self, sample_signals: list[Signal]) -> None:
        coll = SignalCollection(sample_signals)
        bearish = coll.by_direction(SignalDirection.BEARISH)
        assert len(bearish) == 2

    def test_above_strength(self, sample_signals: list[Signal]) -> None:
        coll = SignalCollection(sample_signals)
        strong = coll.above_strength(0.8)
        assert all(s.strength >= 0.8 for s in strong)

    def test_above_confidence(self, sample_signals: list[Signal]) -> None:
        coll = SignalCollection(sample_signals)
        confident = coll.above_confidence(0.85)
        assert all(s.confidence >= 0.85 for s in confident)

    def test_between(self, sample_signals: list[Signal]) -> None:
        coll = SignalCollection(sample_signals)
        start = datetime(2024, 10, 1, tzinfo=UTC)
        end = datetime(2024, 12, 1, tzinfo=UTC)
        filtered = coll.between(start, end)
        assert len(filtered) == 4

    def test_risk_changes(self, sample_signals: list[Signal]) -> None:
        coll = SignalCollection(sample_signals)
        risks = coll.risk_changes()
        assert len(risks) == 1

    def test_risk_changes_by_severity(self, sample_signals: list[Signal]) -> None:
        coll = SignalCollection(sample_signals)
        high_risks = coll.risk_changes(severity="HIGH")
        assert len(high_risks) == 1

    def test_to_dicts(self, sample_signals: list[Signal]) -> None:
        coll = SignalCollection(sample_signals)
        dicts = coll.to_dicts()
        assert len(dicts) == 4
        assert all(isinstance(d, dict) for d in dicts)
        assert dicts[0]["ticker"] == "AAPL"

    def test_chained_filtering(self, sample_signals: list[Signal]) -> None:
        coll = SignalCollection(sample_signals)
        result = coll.by_ticker("AAPL").by_direction("bearish").above_strength(0.5)
        assert len(result) == 1
        assert result[0].signal_type == SignalType.RISK_CHANGE

    def test_repr(self, sample_signals: list[Signal]) -> None:
        coll = SignalCollection(sample_signals)
        assert "count=4" in repr(coll)

    def test_empty_collection(self) -> None:
        coll = SignalCollection()
        assert len(coll) == 0
        assert list(coll) == []
        assert repr(coll) == "SignalCollection(count=0)"

    def test_none_init(self) -> None:
        coll = SignalCollection(None)
        assert len(coll) == 0

    def test_by_direction_string(self, sample_signals: list[Signal]) -> None:
        coll = SignalCollection(sample_signals)
        bearish = coll.by_direction("bearish")
        assert len(bearish) == 2
        assert all(s.direction == SignalDirection.BEARISH for s in bearish)

    def test_between_excludes_outside_range(self, sample_signals: list[Signal]) -> None:
        coll = SignalCollection(sample_signals)
        # Range that excludes all signals
        start = datetime(2020, 1, 1, tzinfo=UTC)
        end = datetime(2020, 12, 31, tzinfo=UTC)
        result = coll.between(start, end)
        assert len(result) == 0

    def test_risk_changes_low_severity_includes_all(
        self, sample_signals: list[Signal]
    ) -> None:
        coll = SignalCollection(sample_signals)
        low_risks = coll.risk_changes(severity="LOW")
        # "HIGH" severity signal should be included (>= LOW)
        assert len(low_risks) == 1

    def test_risk_changes_critical_severity(self, sample_signals: list[Signal]) -> None:
        coll = SignalCollection(sample_signals)
        critical = coll.risk_changes(severity="CRITICAL")
        # Our sample signal is HIGH, not CRITICAL
        assert len(critical) == 0

    def test_supply_chain_edges(self, sample_signals: list[Signal]) -> None:
        coll = SignalCollection(sample_signals)
        edges = coll.supply_chain_edges()
        assert len(edges) == 1
        assert edges[0].source == "AAPL"
        assert edges[0].target == "TSMC"
        assert edges[0].filing_type == FilingType.TEN_K

    def test_supply_chain_edges_preserve_filing_type(self) -> None:
        signal = Signal(
            timestamp=datetime(2024, 11, 1, tzinfo=UTC),
            ticker="AAPL",
            signal_type=SignalType.SUPPLY_CHAIN,
            direction=SignalDirection.NEUTRAL,
            strength=0.8,
            confidence=0.8,
            context="AAPL partners_with TSMC",
            source_filing="",
            related_tickers=["TSMC"],
            metadata={
                "target": "TSMC",
                "relation": "partners_with",
                "edge_context": "joint development",
                "filing_type": "10-Q",
            },
        )
        edges = SignalCollection([signal]).supply_chain_edges()
        assert edges[0].filing_type == FilingType.TEN_Q

    def test_supply_chain_graph(self, sample_signals: list[Signal]) -> None:
        coll = SignalCollection(sample_signals)
        graph = coll.supply_chain_graph()
        assert graph.edge_count >= 1

    def test_to_parquet(self, sample_signals: list[Signal], tmp_path) -> None:
        coll = SignalCollection(sample_signals)
        path = coll.to_parquet(tmp_path / "test.parquet")
        assert path.exists()

    def test_to_csv(self, sample_signals: list[Signal], tmp_path) -> None:
        coll = SignalCollection(sample_signals)
        path = coll.to_csv(tmp_path / "test.csv")
        assert path.exists()

    def test_above_strength_boundary(self, sample_signals: list[Signal]) -> None:
        coll = SignalCollection(sample_signals)
        # Exact boundary: should include signals with exactly that strength
        exact = coll.above_strength(0.95)
        assert all(s.strength >= 0.95 for s in exact)

    def test_above_confidence_boundary(self, sample_signals: list[Signal]) -> None:
        coll = SignalCollection(sample_signals)
        exact = coll.above_confidence(0.92)
        assert all(s.confidence >= 0.92 for s in exact)


class TestPearson:
    """Tests for the _pearson helper."""

    def test_perfect_positive(self) -> None:
        assert _pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)

    def test_perfect_negative(self) -> None:
        assert _pearson([1, 2, 3], [6, 4, 2]) == pytest.approx(-1.0)

    def test_zero_variance_returns_zero(self) -> None:
        assert _pearson([5, 5, 5], [1, 2, 3]) == 0.0

    def test_single_element_returns_zero(self) -> None:
        assert _pearson([1.0], [2.0]) == 0.0


class TestCorrelationMatrix:
    """Tests for CorrelationMatrix."""

    def test_get_method(self) -> None:
        cm = CorrelationMatrix(
            signal_types=["a", "b"],
            matrix=[[1.0, 0.5], [0.5, 1.0]],
            ticker_count=3,
        )
        assert cm.get("a", "b") == 0.5
        assert cm.get("b", "a") == 0.5
        assert cm.get("a", "a") == 1.0


class TestCorrelate:
    """Tests for SignalCollection.correlate()."""

    @staticmethod
    def _sig(
        ticker: str,
        signal_type: SignalType,
        strength: float,
    ) -> Signal:
        return Signal(
            timestamp=datetime(2024, 11, 1, tzinfo=UTC),
            ticker=ticker,
            signal_type=signal_type,
            direction=SignalDirection.NEUTRAL,
            strength=strength,
            confidence=0.9,
            context="test",
            source_filing="",
        )

    def test_requires_at_least_two_types(self) -> None:
        coll = SignalCollection()
        with pytest.raises(ValueError, match="at least two"):
            coll.correlate(SignalType.RISK_CHANGE)

    def test_identity_with_insufficient_tickers(self) -> None:
        """With only 1 ticker, diagonal should be 1.0."""
        signals = [
            self._sig("AAPL", SignalType.RISK_CHANGE, 0.8),
            self._sig("AAPL", SignalType.SUPPLY_CHAIN, 0.9),
        ]
        coll = SignalCollection(signals)
        cm = coll.correlate(SignalType.RISK_CHANGE, SignalType.SUPPLY_CHAIN)
        assert cm.ticker_count == 1
        assert cm.get("risk_change", "risk_change") == 1.0
        assert cm.get("supply_chain", "supply_chain") == 1.0

    def test_positive_correlation(self) -> None:
        """Tickers with correlated signal strengths should yield r > 0."""
        signals = [
            # Ticker with high risk_change -> high supply_chain
            self._sig("AAPL", SignalType.RISK_CHANGE, 0.9),
            self._sig("AAPL", SignalType.SUPPLY_CHAIN, 0.85),
            # Ticker with medium risk_change -> medium supply_chain
            self._sig("MSFT", SignalType.RISK_CHANGE, 0.5),
            self._sig("MSFT", SignalType.SUPPLY_CHAIN, 0.55),
            # Ticker with low risk_change -> low supply_chain
            self._sig("GOOGL", SignalType.RISK_CHANGE, 0.2),
            self._sig("GOOGL", SignalType.SUPPLY_CHAIN, 0.15),
        ]
        coll = SignalCollection(signals)
        cm = coll.correlate("risk_change", "supply_chain")
        assert cm.ticker_count == 3
        assert cm.get("risk_change", "supply_chain") > 0.9

    def test_diagonal_is_one(self) -> None:
        signals = [
            self._sig("AAPL", SignalType.RISK_CHANGE, 0.9),
            self._sig("AAPL", SignalType.M_AND_A, 0.3),
            self._sig("MSFT", SignalType.RISK_CHANGE, 0.4),
            self._sig("MSFT", SignalType.M_AND_A, 0.8),
        ]
        coll = SignalCollection(signals)
        cm = coll.correlate(SignalType.RISK_CHANGE, SignalType.M_AND_A)
        assert cm.get("risk_change", "risk_change") == pytest.approx(1.0)
        assert cm.get("m_and_a", "m_and_a") == pytest.approx(1.0)

    def test_three_types(self) -> None:
        """Correlating three types produces a 3x3 matrix."""
        signals = [
            self._sig("AAPL", SignalType.RISK_CHANGE, 0.9),
            self._sig("AAPL", SignalType.SUPPLY_CHAIN, 0.8),
            self._sig("AAPL", SignalType.TONE_SHIFT, 0.7),
            self._sig("MSFT", SignalType.RISK_CHANGE, 0.4),
            self._sig("MSFT", SignalType.SUPPLY_CHAIN, 0.3),
            self._sig("MSFT", SignalType.TONE_SHIFT, 0.2),
        ]
        coll = SignalCollection(signals)
        cm = coll.correlate("risk_change", "supply_chain", "tone_shift")
        assert len(cm.signal_types) == 3
        assert len(cm.matrix) == 3
        assert all(len(row) == 3 for row in cm.matrix)
