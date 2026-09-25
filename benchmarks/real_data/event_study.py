"""Event-study hook: signal-conditioned abnormal returns from a user price CSV.

Scaffold only -- no price data ships with alphasig and no results are
reported.  Usage::

    python -m benchmarks.real_data.event_study \\
        --signals signals.parquet --prices prices.csv \\
        --market-ticker SPY --windows 1 5 20

``prices.csv`` must have columns ``date,ticker,close`` (ISO dates, daily
closes, split/dividend-adjusted).  ``--signals`` is a Parquet file written
by ``SignalCollection.to_parquet`` or a DuckDB database written by the
pipeline (read via ``SignalStore.to_arrow``).

Point-in-time alignment: a signal stamped at UTC instant ``T`` (when its
filing became public: the EDGAR acceptance time, or 17:30 ET on the filing
date when that is unknown) is first tradeable at the close of the first trading
day whose 16:00 America/New_York close is strictly after ``T``.  That close
is the event day ``t0``; the ``k``-day return runs from close ``t0`` to close
``t0 + k``.  Abnormal return = stock log return minus market log return
(or raw return when no ``--market-ticker`` is given).  Bearish signals are
sign-flipped so a positive mean CAR means the signal was directionally
right.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

_NY = ZoneInfo("America/New_York")
_CLOSE = time(16, 0)


def load_prices(path: str | Path) -> dict[str, tuple[list[date], list[float]]]:
    series: dict[str, list[tuple[date, float]]] = defaultdict(list)
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            series[row["ticker"].upper()].append(
                (date.fromisoformat(row["date"]), float(row["close"]))
            )
    out = {}
    for ticker, rows in series.items():
        rows.sort()
        out[ticker] = ([d for d, _ in rows], [c for _, c in rows])
    return out


def event_index(dates: list[date], ts: datetime) -> int | None:
    """Index of the first trading day whose close is strictly after *ts*."""
    local = ts.astimezone(_NY)
    first = local.date() if local.time() < _CLOSE else local.date() + timedelta(days=1)
    i = bisect.bisect_left(dates, first)
    return i if i < len(dates) else None


def _log_ret(closes: list[float], i: int, k: int) -> float | None:
    if i + k >= len(closes):
        return None
    return math.log(closes[i + k] / closes[i])


def abnormal_returns(
    signals: Iterable[dict[str, Any]],
    prices: dict[str, tuple[list[date], list[float]]],
    windows: list[int],
    market: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    mkt = prices.get(market.upper()) if market else None
    for s in signals:
        ticker = s["ticker"].upper()
        if ticker not in prices:
            continue
        ts = s["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        dates, closes = prices[ticker]
        i = event_index(dates, ts)
        if i is None:
            continue
        sign = -1.0 if s["direction"] == "bearish" else 1.0
        row: dict[str, Any] = {
            "ticker": ticker,
            "signal_type": s["signal_type"],
            "direction": s["direction"],
            "event_date": dates[i].isoformat(),
        }
        for k in windows:
            r = _log_ret(closes, i, k)
            if r is not None and mkt is not None:
                j = bisect.bisect_left(mkt[0], dates[i])
                if j < len(mkt[0]) and mkt[0][j] == dates[i]:
                    m = _log_ret(mkt[1], j, k)
                    r = None if m is None else r - m
                else:
                    r = None
            row[f"car_{k}"] = None if r is None else sign * r
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]], windows: list[int]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["signal_type"], r["direction"])].append(r)
    out = []
    for (stype, direction), grp in sorted(groups.items()):
        entry: dict[str, Any] = {
            "signal_type": stype,
            "direction": direction,
            "n": len(grp),
        }
        for k in windows:
            vals = [r[f"car_{k}"] for r in grp if r[f"car_{k}"] is not None]
            if len(vals) >= 2:
                mean = statistics.fmean(vals)
                sd = statistics.stdev(vals)
                entry[f"mean_car_{k}"] = mean
                entry[f"t_{k}"] = mean / (sd / math.sqrt(len(vals))) if sd > 0 else None
            entry[f"n_{k}"] = len(vals)
        out.append(entry)
    return out


def _load_signals(path: str) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    if path.endswith((".duckdb", ".db")):
        from alphasig.storage import SignalStore

        store = SignalStore(path)
        try:
            table = store.to_arrow(limit=None)
        finally:
            store.close()
    else:
        table = pq.read_table(path)
    return [dict(r) for r in table.to_pylist()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--signals", required=True)
    ap.add_argument("--prices", required=True, help="CSV with date,ticker,close")
    ap.add_argument("--market-ticker", default=None)
    ap.add_argument("--windows", nargs="+", type=int, default=[1, 5, 20])
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if not Path(args.prices).exists():
        print(f"price file not found: {args.prices}", file=sys.stderr)
        return 2
    rows = abnormal_returns(
        _load_signals(args.signals),
        load_prices(args.prices),
        args.windows,
        args.market_ticker,
    )
    result = {"events": len(rows), "summary": summarize(rows, args.windows)}
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
