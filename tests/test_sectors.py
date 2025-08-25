"""Tests for sigint.sectors -- sector classification."""

from __future__ import annotations

from datetime import UTC, datetime

from sigint.models import Signal, SignalDirection, SignalType
from sigint.sectors import Sector, classify_sector
from sigint.signals import SignalCollection


class TestClassifySector:
    """Tests for the classify_sector function."""

    def test_known_tech_ticker(self) -> None:
        assert classify_sector("AAPL") == Sector.TECHNOLOGY
        assert classify_sector("MSFT") == Sector.TECHNOLOGY
        assert classify_sector("NVDA") == Sector.TECHNOLOGY

    def test_known_healthcare_ticker(self) -> None:
        assert classify_sector("UNH") == Sector.HEALTHCARE
        assert classify_sector("JNJ") == Sector.HEALTHCARE

    def test_known_energy_ticker(self) -> None:
        assert classify_sector("XOM") == Sector.ENERGY
        assert classify_sector("CVX") == Sector.ENERGY

    def test_known_financials_ticker(self) -> None:
        assert classify_sector("JPM") == Sector.FINANCIALS
        assert classify_sector("GS") == Sector.FINANCIALS

    def test_unknown_ticker_returns_unknown(self) -> None:
        assert classify_sector("ZZZZZ") == Sector.UNKNOWN

    def test_case_insensitive(self) -> None:
        assert classify_sector("aapl") == Sector.TECHNOLOGY
        assert classify_sector("Msft") == Sector.TECHNOLOGY

    def test_communication_services(self) -> None:
        assert classify_sector("GOOGL") == Sector.COMMUNICATION_SERVICES
        assert classify_sector("META") == Sector.COMMUNICATION_SERVICES

    def test_consumer_discretionary(self) -> None:
        assert classify_sector("AMZN") == Sector.CONSUMER_DISCRETIONARY
        assert classify_sector("TSLA") == Sector.CONSUMER_DISCRETIONARY

    def test_consumer_staples(self) -> None:
        assert classify_sector("WMT") == Sector.CONSUMER_STAPLES
        assert classify_sector("KO") == Sector.CONSUMER_STAPLES

    def test_lookup_table_has_at_least_100_entries(self) -> None:
        from sigint.sectors import _TICKER_SECTOR_MAP

        assert len(_TICKER_SECTOR_MAP) >= 100


class TestSectorEnum:
    """Tests for the Sector enum."""

    def test_all_sectors_have_string_values(self) -> None:
        for sector in Sector:
            assert isinstance(sector.value, str)

    def test_sector_count(self) -> None:
        # 11 GICS-inspired sectors + UNKNOWN
        assert len(Sector) == 12


class TestSignalCollectionBySector:
    """Tests for SignalCollection.by_sector()."""

    def test_by_sector_filters_correctly(self, sample_signals: list[Signal]) -> None:
        coll = SignalCollection(sample_signals)
        tech = coll.by_sector(Sector.TECHNOLOGY)
        # AAPL (2 signals) + MSFT (1 signal) = 3 tech signals
        assert len(tech) == 3
        assert all(s.ticker in {"AAPL", "MSFT"} for s in tech)

    def test_by_sector_string(self, sample_signals: list[Signal]) -> None:
        coll = SignalCollection(sample_signals)
        tech = coll.by_sector("technology")
        assert len(tech) == 3

    def test_by_sector_communication_services(
        self, sample_signals: list[Signal]
    ) -> None:
        coll = SignalCollection(sample_signals)
        comms = coll.by_sector(Sector.COMMUNICATION_SERVICES)
        # META is communication_services
        assert len(comms) == 1
        assert comms[0].ticker == "META"

    def test_by_sector_empty_for_unrepresented(
        self, sample_signals: list[Signal]
    ) -> None:
        coll = SignalCollection(sample_signals)
        energy = coll.by_sector(Sector.ENERGY)
        assert len(energy) == 0

    def test_by_sector_chains_with_other_filters(self) -> None:
        signals = [
            Signal(
                timestamp=datetime(2024, 11, 1, tzinfo=UTC),
                ticker="AAPL",
                signal_type=SignalType.RISK_CHANGE,
                direction=SignalDirection.BEARISH,
                strength=0.8,
                confidence=0.9,
                context="test",
                source_filing="",
            ),
            Signal(
                timestamp=datetime(2024, 11, 1, tzinfo=UTC),
                ticker="MSFT",
                signal_type=SignalType.RISK_CHANGE,
                direction=SignalDirection.BULLISH,
                strength=0.6,
                confidence=0.7,
                context="test",
                source_filing="",
            ),
        ]
        coll = SignalCollection(signals)
        result = coll.by_sector("technology").by_direction("bearish")
        assert len(result) == 1
        assert result[0].ticker == "AAPL"
