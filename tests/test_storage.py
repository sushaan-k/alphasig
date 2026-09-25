"""Tests for alphasig.storage -- DuckDB signal store."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from alphasig.models import Signal, SignalDirection, SignalType
from alphasig.storage import SignalStore


class TestSignalStore:
    """Tests for the SignalStore class."""

    @pytest.fixture
    def store(self) -> SignalStore:
        """Create an in-memory store for each test."""
        return SignalStore(":memory:")

    def test_insert_and_count(
        self, store: SignalStore, sample_signals: list[Signal]
    ) -> None:
        inserted = store.insert(sample_signals)
        assert inserted == 4
        assert store.count() == 4

    def test_query_all(self, store: SignalStore, sample_signals: list[Signal]) -> None:
        store.insert(sample_signals)
        results = store.query()
        assert len(results) == 4

    def test_query_by_ticker(
        self, store: SignalStore, sample_signals: list[Signal]
    ) -> None:
        store.insert(sample_signals)
        results = store.query(ticker="AAPL")
        assert len(results) == 2
        assert all(r.ticker == "AAPL" for r in results)

    def test_query_by_signal_type(
        self, store: SignalStore, sample_signals: list[Signal]
    ) -> None:
        store.insert(sample_signals)
        results = store.query(signal_type="supply_chain")
        assert len(results) == 1

    def test_query_by_direction(
        self, store: SignalStore, sample_signals: list[Signal]
    ) -> None:
        store.insert(sample_signals)
        results = store.query(direction="bearish")
        assert len(results) == 2

    def test_query_by_min_strength(
        self, store: SignalStore, sample_signals: list[Signal]
    ) -> None:
        store.insert(sample_signals)
        results = store.query(min_strength=0.8)
        assert all(r.strength >= 0.8 for r in results)

    def test_query_with_limit(
        self, store: SignalStore, sample_signals: list[Signal]
    ) -> None:
        store.insert(sample_signals)
        results = store.query(limit=2)
        assert len(results) == 2

    def test_summary(self, store: SignalStore, sample_signals: list[Signal]) -> None:
        store.insert(sample_signals)
        summary = store.summary()
        assert summary["total"] == 4
        assert len(summary["by_type_direction"]) > 0

    def test_empty_store(self, store: SignalStore) -> None:
        assert store.count() == 0
        results = store.query()
        assert results == []

    def test_roundtrip_preserves_data(
        self, store: SignalStore, sample_signals: list[Signal]
    ) -> None:
        store.insert(sample_signals)
        results = store.query(ticker="MSFT")
        assert len(results) == 1
        sig = results[0]
        assert sig.signal_type == SignalType.M_AND_A
        assert sig.direction == SignalDirection.BULLISH
        assert sig.metadata.get("indicator_count") == 2

    def test_query_by_min_confidence(
        self, store: SignalStore, sample_signals: list[Signal]
    ) -> None:
        store.insert(sample_signals)
        results = store.query(min_confidence=0.85)
        assert all(r.confidence >= 0.85 for r in results)

    def test_query_by_time_range(
        self, store: SignalStore, sample_signals: list[Signal]
    ) -> None:
        store.insert(sample_signals)
        start = datetime(2024, 10, 1, tzinfo=UTC)
        end = datetime(2024, 12, 1, tzinfo=UTC)
        results = store.query(start=start, end=end)
        assert len(results) == 4

    def test_query_by_time_range_excludes(
        self, store: SignalStore, sample_signals: list[Signal]
    ) -> None:
        store.insert(sample_signals)
        start = datetime(2020, 1, 1, tzinfo=UTC)
        end = datetime(2020, 12, 31, tzinfo=UTC)
        results = store.query(start=start, end=end)
        assert len(results) == 0

    def test_roundtrip_preserves_utc_timestamp(
        self, store: SignalStore, sample_signals: list[Signal]
    ) -> None:
        expected = sample_signals[0].timestamp
        store.insert([sample_signals[0]])
        results = store.query(ticker=sample_signals[0].ticker)
        assert results[0].timestamp == expected
        assert results[0].timestamp.tzinfo == UTC

    def test_multiple_inserts_accumulate(
        self, store: SignalStore, sample_signals: list[Signal]
    ) -> None:
        store.insert(sample_signals[:2])
        store.insert(sample_signals[2:])
        assert store.count() == 4

    def test_close_and_reopen(self, sample_signals: list[Signal], tmp_path) -> None:
        db_path = str(tmp_path / "test.duckdb")
        store1 = SignalStore(db_path)
        store1.insert(sample_signals)
        store1.close()

        store2 = SignalStore(db_path)
        assert store2.count() == 4
        store2.close()

    def test_summary_empty_store(self, store: SignalStore) -> None:
        summary = store.summary()
        assert summary["total"] == 0
        assert summary["by_type_direction"] == []

    def test_query_combined_filters(
        self, store: SignalStore, sample_signals: list[Signal]
    ) -> None:
        store.insert(sample_signals)
        results = store.query(
            ticker="AAPL",
            direction="bearish",
            min_strength=0.5,
        )
        assert len(results) == 1
        assert results[0].signal_type == SignalType.RISK_CHANGE


def _sig(i: int, **overrides: object) -> Signal:
    fields: dict[str, object] = {
        "timestamp": datetime(2024, 11, 1, 22, 4, tzinfo=UTC),
        "ticker": "AAPL",
        "signal_type": SignalType.RISK_CHANGE,
        "direction": SignalDirection.BEARISH,
        "strength": 0.5,
        "confidence": 0.8,
        "context": f"risk {i}",
        "source_filing": "https://www.sec.gov/Archives/x.htm",
    }
    fields.update(overrides)
    return Signal(**fields)  # type: ignore[arg-type]


class TestSignalStoreRegressions:
    def test_reinserting_same_signals_is_idempotent(self) -> None:
        with SignalStore(":memory:") as store:
            batch = [_sig(1), _sig(2)]
            assert store.insert(batch) == 2
            assert store.insert([*batch, _sig(3), _sig(3)]) == 1
            assert store.count() == 3

    def test_empty_insert(self) -> None:
        with SignalStore(":memory:") as store:
            assert store.insert([]) == 0

    def test_aware_bounds_do_not_depend_on_session_timezone(self) -> None:
        from datetime import timedelta, timezone

        with SignalStore(":memory:") as store:
            store._conn.execute("SET TimeZone = 'America/New_York'")
            store.insert([_sig(1)])
            # Same instant as the stored signal, expressed at UTC-5.
            bound = datetime(2024, 11, 1, 17, 4, tzinfo=timezone(timedelta(hours=-5)))
            assert len(store.query(start=bound)) == 1
            assert len(store.query(end=bound)) == 1
            assert store.query(start=bound + timedelta(seconds=1)) == []

    def test_round_trip_preserves_utc_instant(self) -> None:
        with SignalStore(":memory:") as store:
            store.insert([_sig(1)])
            (out,) = store.query()
            assert out.timestamp == datetime(2024, 11, 1, 22, 4, tzinfo=UTC)
