# Backtesting Guide

alphasig outputs structured, timestamped signals designed for integration with quantitative backtesting frameworks.

## Parquet Export

The primary export format for backtesting is Parquet, which pandas, polars and DuckDB read directly.

```python
from alphasig import Pipeline

pipeline = Pipeline(user_agent="Your Name your@email.com")
signals = await pipeline.extract(
    tickers=["AAPL", "MSFT"],
    filing_types=["10-K", "10-Q"],
    lookback_years=5,
)

# Export all signals
signals.to_parquet("all_signals.parquet")

# Export only high-conviction bearish signals
bearish = signals.by_direction("bearish").above_strength(0.7)
bearish.to_parquet("bearish_signals.parquet")
```

## Using with pandas

```python
import pandas as pd

df = pd.read_parquet("all_signals.parquet")

# Bucket by the US/Eastern calendar day the filing became public
df["day"] = df["timestamp"].dt.tz_convert("America/New_York").dt.normalize()
daily = df.pivot_table(
    index="day",
    columns="ticker",
    values="strength",
    aggfunc="mean",
)
```

Signals already stored in DuckDB load straight into a DataFrame (needs
`pip install "alphasig[pandas]"`; `to_arrow()` needs only pyarrow):

```python
from alphasig import SignalStore

with SignalStore("alphasig.duckdb") as store:
    df = store.to_pandas(min_confidence=0.8)
```

Filings accepted after the 16:00 ET close are only tradeable at the next
session; shift those rows forward before joining to daily bars:

```python
after_close = df["timestamp"].dt.tz_convert("America/New_York").dt.hour >= 16
df.loc[after_close, "day"] += pd.offsets.BDay(1)
```

## Signal Timing

Signal timestamps are the moment the source filing became public: EDGAR's acceptance time, converted to UTC (EDGAR reports it in Eastern time). When the acceptance time is unavailable, 17:30 ET on the filing date is used -- EDGAR assigns filings accepted after 17:30 to the next business day, so that fallback never precedes the real release. The period-of-report date is kept in `metadata["_period_of_report"]` but is never used as the timestamp.

Amendments (`10-K/A`, `10-Q/A`) are not fetched: the original filing is what the market saw first. If the same signal does appear twice within a filing family, the earliest copy is kept.

When ranking historically, pass `as_of` (or `--as-of`): signals published after that time are excluded rather than scored.

## Combining Signal Types

Different signal types can be combined into a composite score:

```python
# Weight signals by type
weights = {
    "risk_change": 0.35,
    "tone_shift": 0.25,
    "supply_chain": 0.20,
    "m_and_a": 0.20,
}

for signal in signals:
    w = weights.get(signal.signal_type.value, 0.25)
    composite = signal.strength * w
```

## Ranking Signals Offline

For portfolio review or pre-backtest screening, score the local signal store
without re-running EDGAR or LLM extraction:

```bash
alphasig rank --db alphasig.duckdb --min-confidence 0.8 --limit 25
alphasig rank --db alphasig.duckdb --format json --output ranking.json
alphasig rank --db alphasig.duckdb --as-of 2025-01-15T00:00:00Z
alphasig rank --db alphasig.duckdb --as-of 2025-01-15T00:00:00Z --half-life 90
```

The ranking uses confidence-weighted directional strength:

- bullish signals contribute positive exposure
- bearish signals contribute negative exposure
- neutral signals contribute gross exposure but not net direction
- `--as-of` scores the store as of that time: later signals are excluded and
  each signal's configured `decay_rate` is applied
- `--half-life DAYS` decays every signal with that half-life instead
  (implies `--as-of` now when not given)

The same logic is available from Python:

```python
from alphasig import SignalStore, rank_signals

store = SignalStore("alphasig.duckdb")
signals = store.query(min_confidence=0.8, limit=100_000)
store.close()

report = rank_signals(signals, limit=25)
print(report.to_json())
```

For portfolio construction, summarize the same directional exposure by sector:

```bash
alphasig sectors --db alphasig.duckdb --min-confidence 0.8 --limit 5
alphasig sectors --db alphasig.duckdb --exclude-unknown --format markdown \
  --output sector_exposure.md
```

```python
from alphasig import SignalStore, summarize_sector_exposure

store = SignalStore("alphasig.duckdb")
signals = store.query(min_confidence=0.8, limit=100_000)
store.close()

report = summarize_sector_exposure(signals, limit=5, include_unknown=False)
print(report.to_markdown())
```

Sector scores use the same confidence-weighted, optionally decayed signal
strength as ticker ranking, then group tickers with the built-in sector map.
This helps catch concentrated bearish or bullish exposure before allocating
capital to a strategy slice.

## DuckDB Analytics

For more complex queries, use the DuckDB store directly:

```python
from alphasig import SignalStore

store = SignalStore("alphasig.duckdb")

# Get summary statistics
summary = store.summary()

# Custom queries
signals = store.query(
    ticker="AAPL",
    signal_type="risk_change",
    min_strength=0.5,
)
```
