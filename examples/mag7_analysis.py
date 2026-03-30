"""Analyse the Magnificent 7 for causal signals.

Extracts supply-chain dependencies, risk-factor changes, M&A indicators,
and management tone shifts from the latest 10-K and 10-Q filings of
AAPL, MSFT, GOOGL, META, AMZN, NVDA, and TSLA.

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python examples/mag7_analysis.py
"""

from __future__ import annotations

import asyncio

from sigint import Pipeline, SignalCollection

MAG7 = ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "TSLA"]


async def main() -> None:
    pipeline = Pipeline(
        model="claude-sonnet-4-6",
        user_agent="sigint-example research@example.com",
        cache_dir="./edgar_cache",
        db_path="mag7.duckdb",
    )

    print(f"Extracting signals for {', '.join(MAG7)}...")
    signals: SignalCollection = await pipeline.extract(
        tickers=MAG7,
        filing_types=["10-K", "10-Q"],
        lookback_years=2,
        engines=["supply_chain", "risk_differ", "m_and_a", "tone"],
    )

    print(f"\nTotal signals extracted: {len(signals)}")

    # Breakdown by type
    for stype in ["supply_chain", "risk_change", "m_and_a", "tone_shift"]:
        subset = signals.by_type(stype)
        print(f"  {stype}: {len(subset)}")

    # Show high-conviction bearish signals
    bearish = signals.by_direction("bearish").above_strength(0.7)
    print(f"\nHigh-conviction bearish signals: {len(bearish)}")
    for sig in bearish:
        print(f"  [{sig.ticker}] {sig.context[:80]}")

    # Export
    signals.to_parquet("mag7_signals.parquet")
    signals.to_csv("mag7_signals.csv")
    print("\nExported to mag7_signals.parquet and mag7_signals.csv")


if __name__ == "__main__":
    asyncio.run(main())
