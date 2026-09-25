"""Tests for alphasig.storage -- DuckDB signal store."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

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


class TestArrowExport:
    """SignalStore.to_arrow / to_pandas."""

    @pytest.fixture
    def store(self, sample_signals: list[Signal]) -> SignalStore:
        s = SignalStore(":memory:")
        s.insert(sample_signals)
        return s

    def test_to_arrow_matches_parquet_schema(self, store: SignalStore) -> None:
        from alphasig.output.parquet import _SCHEMA

        table = store.to_arrow()
        assert table.schema == _SCHEMA
        assert table.num_rows == 4

    def test_to_arrow_filters_and_limit(self, store: SignalStore) -> None:
        assert store.to_arrow(ticker="aapl").num_rows == 2
        assert store.to_arrow(signal_type="m_and_a").column("ticker").to_pylist() == [
            "MSFT"
        ]
        assert store.to_arrow(min_confidence=0.9).num_rows == 1
        assert store.to_arrow(limit=1).num_rows == 1

    def test_to_arrow_agrees_with_query(self, store: SignalStore) -> None:
        import json

        rows = store.to_arrow(ticker="AAPL").to_pylist()
        signals = store.query(ticker="AAPL")
        assert [r["timestamp"] for r in rows] == [s.timestamp for s in signals]
        assert [json.loads(r["metadata"]) for r in rows] == [
            s.metadata for s in signals
        ]
        assert rows[0]["timestamp"].tzinfo is not None

    def test_to_arrow_time_range_ignores_session_time_zone(self) -> None:
        with SignalStore(":memory:") as store:
            store._conn.execute("SET TimeZone = 'America/New_York'")
            store.insert([_sig(1)])
            instant = datetime(2024, 11, 1, 22, 4, tzinfo=UTC)
            assert store.to_arrow(start=instant, end=instant).num_rows == 1
            (row,) = store.to_arrow().to_pylist()
            assert row["timestamp"] == instant

    def test_to_arrow_empty_store(self) -> None:
        with SignalStore(":memory:") as store:
            table = store.to_arrow()
            assert table.num_rows == 0
            assert "timestamp" in table.schema.names

    def test_to_pandas_without_pandas_is_a_clear_error(
        self, store: SignalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import importlib.util

        real = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name, *a: None if name == "pandas" else real(name, *a),
        )
        with pytest.raises(ImportError, match=r"alphasig\[pandas\]"):
            store.to_pandas()

    def test_to_pandas(self, store: SignalStore) -> None:
        pytest.importorskip("pandas")
        df = store.to_pandas(direction="bearish")
        assert len(df) == 2
        assert str(df["timestamp"].dt.tz) == "UTC"


class TestExtractionLog:
    """The extraction_log behind Pipeline.extract(incremental=True)."""

    @staticmethod
    def _job(accession: str, engine: str = "risk_differ", **kw: object) -> Any:
        from alphasig.storage import ExtractionRecord

        fields: dict[str, Any] = {
            "accession": accession,
            "engine": engine,
            "ticker": "AAPL",
            "signal_count": 1,
        }
        fields.update(kw)
        return ExtractionRecord(**fields)

    @staticmethod
    def _job_sig(accession: str, i: int) -> Signal:
        return _sig(i, metadata={"_filing_accession": accession})

    def test_record_and_read_back(self) -> None:
        with SignalStore(":memory:") as store:
            jobs = [
                self._job("acc-1", previous_accession="acc-0", calibrated=True),
                self._job("acc-1", "m_and_a", signal_count=0),
            ]
            assert store.record_extraction([self._job_sig("acc-1", 1)], jobs) == 1
            done = store.completed_extractions()
            assert set(done) == {("acc-1", "risk_differ"), ("acc-1", "m_and_a")}
            assert done[("acc-1", "risk_differ")] == jobs[0]
            assert done[("acc-1", "m_and_a")].previous_accession is None
            assert not done[("acc-1", "m_and_a")].calibrated

    def test_rerecording_is_idempotent(self) -> None:
        with SignalStore(":memory:") as store:
            signals = [self._job_sig("acc-1", 1)]
            store.record_extraction(signals, [self._job("acc-1")])
            # Same job again: the log row is overwritten, the signal skipped.
            assert store.record_extraction(signals, [self._job("acc-1")]) == 0
            assert store.count() == 1
            assert len(store.completed_extractions()) == 1

    def test_replace_deletes_the_jobs_previous_signals(self) -> None:
        with SignalStore(":memory:") as store:
            store.record_extraction(
                [
                    self._job_sig("acc-1", 1),
                    self._job_sig("acc-2", 2),
                    _sig(
                        3,
                        signal_type=SignalType.TONE_SHIFT,
                        metadata={"_filing_accession": "acc-1"},
                    ),
                ],
                [self._job("acc-1"), self._job("acc-2")],
            )
            store.record_extraction(
                [self._job_sig("acc-1", 4)],
                [self._job("acc-1", calibrated=True)],
                replace=[("acc-1", "risk_change")],
            )
            contexts = sorted(s.context for s in store.query())
            # acc-1's old risk signal is gone; other jobs are untouched.
            assert contexts == ["risk 2", "risk 3", "risk 4"]
            assert store.completed_extractions()[("acc-1", "risk_differ")].calibrated

    def test_failure_rolls_back_signals_and_log(self) -> None:
        import duckdb

        from alphasig.exceptions import StorageError

        store = SignalStore(":memory:")
        store.record_extraction([self._job_sig("acc-1", 1)], [self._job("acc-1")])
        real = store._conn

        class _FailingLogConn:
            def __getattr__(self, name: str) -> object:
                return getattr(real, name)

            def execute(self, sql: str, *args: object) -> object:
                if "extraction_log" in sql:
                    raise duckdb.Error("disk full")
                return real.execute(sql, *args)

        store._conn = _FailingLogConn()  # type: ignore[assignment]
        with pytest.raises(StorageError):
            store.record_extraction(
                [self._job_sig("acc-2", 2)],
                [self._job("acc-2")],
                replace=[("acc-1", "risk_change")],
            )
        store._conn = real
        # Neither the delete, the insert nor the log row was kept.
        assert sorted(s.context for s in store.query()) == ["risk 1"]
        assert set(store.completed_extractions()) == {("acc-1", "risk_differ")}
        store.close()

    def test_log_survives_reopen(self, tmp_path: Any) -> None:
        path = tmp_path / "log.duckdb"
        with SignalStore(path) as store:
            store.record_extraction([], [self._job("acc-1")])
        with SignalStore(path) as store:
            assert ("acc-1", "risk_differ") in store.completed_extractions()
