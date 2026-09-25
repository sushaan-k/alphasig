"""Point-in-time alignment in the (scaffold) event-study hook."""

from __future__ import annotations

from datetime import UTC, date, datetime

from benchmarks.real_data.event_study import abnormal_returns, event_index

DATES = [date(2024, 10, 31), date(2024, 11, 1), date(2024, 11, 4), date(2024, 11, 5)]


def test_accepted_before_close_trades_same_day() -> None:
    # 15:59 ET on 2024-11-01 (EDT) == 19:59 UTC
    ts = datetime(2024, 11, 1, 19, 59, tzinfo=UTC)
    assert DATES[event_index(DATES, ts) or 0] == date(2024, 11, 1)


def test_accepted_after_close_trades_next_session() -> None:
    # 16:06 ET Friday -> first close after it is Monday 2024-11-04
    ts = datetime(2024, 11, 1, 20, 6, 21, tzinfo=UTC)
    assert DATES[event_index(DATES, ts) or 0] == date(2024, 11, 4)


def test_bearish_signals_are_sign_flipped() -> None:
    prices = {"X": (DATES, [100.0, 100.0, 90.0, 90.0])}
    sig = {
        "ticker": "X",
        "signal_type": "risk_change",
        "direction": "bearish",
        "timestamp": datetime(2024, 11, 1, 19, 0, tzinfo=UTC),
    }
    (row,) = abnormal_returns([sig], prices, windows=[1])
    assert row["event_date"] == "2024-11-01"
    assert row["car_1"] > 0  # price fell after a bearish signal
