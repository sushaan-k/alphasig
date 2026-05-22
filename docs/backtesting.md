# Backtesting Guide

sigint outputs structured, timestamped signals designed for integration with quantitative backtesting frameworks.

## Parquet Export

The primary export format for backtesting is Parquet, which is natively supported by Lean, Zipline, and most pandas-based backtest systems.

```python
from sigint import Pipeline

pipeline = Pipeline(model="claude-sonnet-4-6")
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

# Pivot to get daily signal scores by ticker
daily = df.pivot_table(
    index="timestamp",
    columns="ticker",
    values="strength",
    aggfunc="mean",
)
```

## Signal Timing

All signal timestamps correspond to the SEC filing date (not the period-of-report date). This is the date the information became publicly available, which is the relevant date for backtesting.

Signals are point-in-time: they reflect only information available at the filing date. No look-ahead bias is introduced.

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
sigint rank --db sigint.duckdb --min-confidence 0.8 --limit 25
sigint rank --db sigint.duckdb --format json --output ranking.json
sigint rank --db sigint.duckdb --as-of 2025-01-15T00:00:00Z
```

The ranking uses confidence-weighted directional strength:

- bullish signals contribute positive exposure
- bearish signals contribute negative exposure
- neutral signals contribute gross exposure but not net direction
- `--as-of` evaluates each signal's configured decay before scoring

The same logic is available from Python:

```python
from sigint import SignalStore, rank_signals

store = SignalStore("sigint.duckdb")
signals = store.query(min_confidence=0.8, limit=100_000)
store.close()

report = rank_signals(signals, limit=25)
print(report.to_json())
```

## DuckDB Analytics

For more complex queries, use the DuckDB store directly:

```python
from sigint import SignalStore

store = SignalStore("sigint.duckdb")

# Get summary statistics
summary = store.summary()

# Custom queries
signals = store.query(
    ticker="AAPL",
    signal_type="risk_change",
    min_strength=0.5,
)
```
