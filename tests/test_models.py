"""Tests for sigint.models -- Pydantic data models."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from sigint.models import (
    Filing,
    FilingSection,
    FilingType,
    RelationType,
    RiskChange,
    RiskChangeType,
    Severity,
    Signal,
    SignalDirection,
    SignalType,
    SupplyChainEdge,
    ToneLabel,
    TonePoint,
    ToneTrajectory,
)


class TestSignal:
    """Tests for the Signal model."""

    def test_create_valid_signal(self) -> None:
        signal = Signal(
            timestamp=datetime(2024, 11, 1, tzinfo=UTC),
            ticker="AAPL",
            signal_type=SignalType.SUPPLY_CHAIN,
            direction=SignalDirection.NEUTRAL,
            strength=0.8,
            confidence=0.9,
            context="Test signal",
            source_filing="https://sec.gov/test",
        )
        assert signal.ticker == "AAPL"
        assert signal.strength == 0.8
        assert signal.related_tickers == []
        assert signal.metadata == {}

    def test_strength_rejects_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            Signal(
                timestamp=datetime(2024, 1, 1, tzinfo=UTC),
                ticker="MSFT",
                signal_type=SignalType.RISK_CHANGE,
                direction=SignalDirection.BEARISH,
                strength=1.5,
                confidence=0.5,
                context="Out of range test",
                source_filing="",
            )
        with pytest.raises(ValidationError):
            Signal(
                timestamp=datetime(2024, 1, 1, tzinfo=UTC),
                ticker="MSFT",
                signal_type=SignalType.RISK_CHANGE,
                direction=SignalDirection.BEARISH,
                strength=0.5,
                confidence=-0.2,
                context="Out of range test",
                source_filing="",
            )

    def test_signal_is_frozen(self) -> None:
        signal = Signal(
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            ticker="GOOG",
            signal_type=SignalType.TONE_SHIFT,
            direction=SignalDirection.BULLISH,
            strength=0.5,
            confidence=0.5,
            context="Frozen test",
            source_filing="",
        )
        with pytest.raises(ValidationError):
            signal.ticker = "CHANGED"  # type: ignore[misc]

    def test_signal_serialisation_roundtrip(self) -> None:
        signal = Signal(
            timestamp=datetime(2024, 11, 1, tzinfo=UTC),
            ticker="META",
            signal_type=SignalType.M_AND_A,
            direction=SignalDirection.BULLISH,
            strength=0.85,
            confidence=0.72,
            context="Roundtrip test",
            source_filing="https://sec.gov/test",
            related_tickers=["GOOGL"],
            metadata={"key": "value"},
        )
        data = signal.model_dump(mode="json")
        restored = Signal.model_validate(data)
        assert restored == signal

    def test_signal_rejects_invalid_strength(self) -> None:
        with pytest.raises(ValidationError):
            Signal(
                timestamp=datetime(2024, 1, 1, tzinfo=UTC),
                ticker="X",
                signal_type="invalid_type",  # type: ignore[arg-type]
                direction=SignalDirection.NEUTRAL,
                strength=0.5,
                confidence=0.5,
                context="",
                source_filing="",
            )

    def test_decay_rate_defaults_to_zero(self) -> None:
        signal = Signal(
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            ticker="AAPL",
            signal_type=SignalType.RISK_CHANGE,
            direction=SignalDirection.BEARISH,
            strength=0.8,
            confidence=0.9,
            context="test",
            source_filing="",
        )
        assert signal.decay_rate == 0.0

    def test_current_strength_no_decay(self) -> None:
        """With decay_rate=0 the strength never changes."""
        signal = Signal(
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            ticker="AAPL",
            signal_type=SignalType.RISK_CHANGE,
            direction=SignalDirection.BEARISH,
            strength=0.8,
            confidence=0.9,
            context="test",
            source_filing="",
            decay_rate=0.0,
        )
        future = datetime(2025, 6, 15, tzinfo=UTC)
        assert signal.current_strength(as_of=future) == 0.8

    def test_current_strength_decays_over_time(self) -> None:
        """Exponential decay reduces strength as time passes."""
        signal = Signal(
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            ticker="AAPL",
            signal_type=SignalType.RISK_CHANGE,
            direction=SignalDirection.BEARISH,
            strength=1.0,
            confidence=0.9,
            context="test",
            source_filing="",
            decay_rate=0.01,  # half-life ~69 days
        )
        # After 69 days, strength should be approximately 0.5
        after_69_days = datetime(2024, 3, 10, tzinfo=UTC)  # ~69 days
        decayed = signal.current_strength(as_of=after_69_days)
        assert 0.45 < decayed < 0.55

    def test_current_strength_before_timestamp(self) -> None:
        """Evaluating before the signal timestamp returns original strength."""
        signal = Signal(
            timestamp=datetime(2024, 6, 1, tzinfo=UTC),
            ticker="AAPL",
            signal_type=SignalType.SUPPLY_CHAIN,
            direction=SignalDirection.NEUTRAL,
            strength=0.9,
            confidence=0.8,
            context="test",
            source_filing="",
            decay_rate=0.01,
        )
        before = datetime(2024, 1, 1, tzinfo=UTC)
        assert signal.current_strength(as_of=before) == 0.9

    def test_current_strength_far_future_approaches_zero(self) -> None:
        """After a very long time the signal should be near zero."""
        signal = Signal(
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            ticker="MSFT",
            signal_type=SignalType.TONE_SHIFT,
            direction=SignalDirection.BEARISH,
            strength=0.9,
            confidence=0.8,
            context="test",
            source_filing="",
            decay_rate=0.01,
        )
        far_future = datetime(2030, 1, 1, tzinfo=UTC)
        assert signal.current_strength(as_of=far_future) < 0.01


class TestFiling:
    """Tests for the Filing model."""

    def test_create_filing(self, sample_filing: Filing) -> None:
        assert sample_filing.ticker == "AAPL"
        assert sample_filing.filing_type == FilingType.TEN_K
        assert sample_filing.cik == "0000320193"

    def test_filing_is_hashable(self, sample_filing: Filing) -> None:
        s = {sample_filing}
        assert len(s) == 1


class TestFilingSection:
    """Tests for the FilingSection model."""

    def test_create_section(self) -> None:
        section = FilingSection(
            filing_accession="0001234567-24-000001",
            ticker="MSFT",
            section_name="Risk Factors",
            section_key="risk_factors",
            text="Some risk factor text...",
            filing_type=FilingType.TEN_Q,
            filed_date=date(2024, 7, 30),
        )
        assert section.section_key == "risk_factors"
        assert section.filing_type == FilingType.TEN_Q


class TestSupplyChainEdge:
    """Tests for the SupplyChainEdge model."""

    def test_create_edge(self) -> None:
        edge = SupplyChainEdge(
            source="AAPL",
            target="TSMC",
            relation=RelationType.DEPENDS_ON,
            context="semiconductor manufacturing",
            confidence=0.95,
            filing_type=FilingType.TEN_K,
            filed_date=date(2024, 11, 1),
        )
        assert edge.source == "AAPL"
        assert edge.relation == RelationType.DEPENDS_ON

    def test_edge_confidence_must_be_unit(self) -> None:
        with pytest.raises(ValidationError):
            SupplyChainEdge(
                source="A",
                target="B",
                relation=RelationType.DEPENDS_ON,
                context="",
                confidence=2.0,
                filing_type=FilingType.TEN_K,
                filed_date=date(2024, 1, 1),
            )


class TestRiskChange:
    """Tests for the RiskChange model."""

    def test_create_risk_change(self) -> None:
        change = RiskChange(
            company="MSFT",
            ticker="MSFT",
            change_type=RiskChangeType.NEW,
            risk="AI infrastructure spending concentration",
            current_filing="10-K 2025",
            severity_estimate=Severity.HIGH,
            related_tickers=["NVDA"],
        )
        assert change.change_type == RiskChangeType.NEW
        assert change.severity_estimate == Severity.HIGH


class TestToneModels:
    """Tests for TonePoint and ToneTrajectory."""

    def test_tone_point(self) -> None:
        pt = TonePoint(
            filing_label="10-Q Q3 2025",
            tone=ToneLabel.HEDGING_CAUTIOUS,
            confidence=0.71,
        )
        assert pt.tone == ToneLabel.HEDGING_CAUTIOUS

    def test_tone_trajectory(self) -> None:
        traj = ToneTrajectory(
            company="META",
            ticker="META",
            topic="AI infrastructure spending",
            trajectory=[
                TonePoint(
                    filing_label="10-Q Q1 2025",
                    tone=ToneLabel.CONFIDENT_EXPANDING,
                    confidence=0.85,
                ),
                TonePoint(
                    filing_label="10-Q Q2 2025",
                    tone=ToneLabel.HEDGING_CAUTIOUS,
                    confidence=0.71,
                ),
            ],
            signal=SignalDirection.BEARISH,
            signal_strength=0.78,
        )
        assert len(traj.trajectory) == 2
        assert traj.signal == SignalDirection.BEARISH


class TestEnums:
    """Tests for enum types."""

    def test_filing_types(self) -> None:
        assert FilingType.TEN_K.value == "10-K"
        assert FilingType.TEN_Q.value == "10-Q"
        assert FilingType.EIGHT_K.value == "8-K"
        assert FilingType.DEF_14A.value == "DEF 14A"

    def test_signal_types(self) -> None:
        assert len(SignalType) == 4

    def test_tone_labels(self) -> None:
        assert len(ToneLabel) == 6
