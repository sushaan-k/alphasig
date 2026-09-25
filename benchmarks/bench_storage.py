"""DuckDB SignalStore insert / query latency at 10k, 100k and 1M signals.

Also times re-inserting the same batch (all duplicates, skipped by the
idempotent insert) and the ``to_arrow()`` export when the store has one.

Signals are synthetic (seeded RNG) with realistic cardinalities: 3,000
tickers, 4 signal types, 3 directions, ~15 years of timestamps.  Each size
uses a fresh on-disk database in a temporary directory.
"""

from __future__ import annotations

import inspect
import random
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from alphasig.models import Signal, SignalDirection, SignalType
from alphasig.storage import SignalStore
from benchmarks._common import SEED

_TYPES = list(SignalType)
_DIRS = list(SignalDirection)


def make_signals(n: int, seed: int = SEED) -> list[Signal]:
    rng = random.Random(seed)
    base = datetime(2010, 1, 1, tzinfo=UTC)
    out = []
    for i in range(n):
        out.append(
            Signal(
                timestamp=base + timedelta(minutes=rng.randrange(15 * 365 * 24 * 60)),
                ticker=f"T{rng.randrange(3000):04d}",
                signal_type=_TYPES[i % 4],
                direction=_DIRS[rng.randrange(3)],
                strength=rng.random(),
                confidence=rng.random(),
                context=f"ESCALATED: synthetic risk change {i}",
                source_filing=f"https://www.sec.gov/Archives/edgar/data/{i}.htm",
                related_tickers=[f"T{rng.randrange(3000):04d}"],
                metadata={"severity": "HIGH", "i": i},
            )
        )
    return out


def _t(fn: Any) -> tuple[float, Any]:
    t0 = time.perf_counter()
    res = fn()
    return time.perf_counter() - t0, res


def _query_suite(store: SignalStore, repeat: int) -> dict[str, float]:
    t_mid = datetime(2017, 6, 1, tzinfo=UTC)
    cases: dict[str, Any] = {
        "ticker_eq": lambda: store.query(ticker="T0042", limit=1000),
        "type_and_min_strength": lambda: store.query(
            signal_type="risk_change", min_strength=0.9, limit=1000
        ),
        "time_range_30d": lambda: store.query(
            start=t_mid, end=t_mid + timedelta(days=30), limit=1000
        ),
        "latest_1000": lambda: store.query(limit=1000),
        "count": store.count,
        "summary": store.summary,
    }
    if hasattr(store, "to_arrow"):
        cases["to_arrow_ticker_eq"] = lambda: store.to_arrow(ticker="T0042", limit=1000)
        cases["to_arrow_all"] = lambda: store.to_arrow(limit=None)
    out = {}
    for name, fn in cases.items():
        fn()  # warm
        samples = sorted(_t(fn)[0] for _ in range(repeat))
        out[name + "_ms"] = round(samples[len(samples) // 2] * 1000, 3)
    return out


def run(quick: bool = False, sizes: tuple[int, ...] | None = None) -> dict[str, Any]:
    if sizes is None:
        sizes = (10_000,) if quick else (10_000, 100_000, 1_000_000)
    repeat = 3 if quick else 7
    rows = []
    # Row-by-row executemany is too slow to run at 100k+; it is measured only
    # when the store still uses it (baseline) and only at the smallest size.
    src = inspect.getsource(SignalStore.insert)
    uses_executemany = "executemany" in src
    for n in sizes:
        if uses_executemany and n > 10_000:
            rows.append({"signals": n, "skipped": "row-by-row insert too slow"})
            continue
        signals = make_signals(n)
        with tempfile.TemporaryDirectory() as tmp:
            store = SignalStore(Path(tmp) / "bench.duckdb")
            dt, _ = _t(lambda s=store, sig=signals: s.insert(sig))
            row: dict[str, Any] = {
                "signals": n,
                "insert_s": round(dt, 4),
                "insert_rows_per_s": round(n / dt),
            }
            if not uses_executemany:
                # Re-inserting the same batch: every row is a duplicate and is
                # skipped by the idempotent insert.
                dt_re, again = _t(lambda s=store, sig=signals: s.insert(sig))
                row["reinsert_same_s"] = round(dt_re, 4)
                row["reinsert_inserted"] = again
            row.update(_query_suite(store, repeat))
            dt_q, res = _t(lambda s=store, lim=n: s.query(limit=lim))
            row["query_all_to_signals_s"] = round(dt_q, 4)
            assert len(res) == n
            store.close()
            rows.append(row)
    return {
        "insert_path": "executemany" if uses_executemany else "arrow",
        "sizes": rows,
    }
