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
