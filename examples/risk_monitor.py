"""Monitor risk-factor changes across companies.

Diffs risk factors between the two most recent 10-K filings for a set
of companies and highlights material changes.  Useful for weekly
monitoring of portfolio risk exposure.

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python examples/risk_monitor.py
"""

from __future__ import annotations

import asyncio

from sigint import Pipeline, SignalCollection

WATCHLIST = ["AAPL", "MSFT", "GOOGL", "META", "AMZN"]


async def main() -> None:
    pipeline = Pipeline(
        model="claude-sonnet-4-6",
        user_agent="sigint-example research@example.com",
        cache_dir="./edgar_cache",
        db_path="risk_monitor.duckdb",
    )

    print(f"Monitoring risk factors for {', '.join(WATCHLIST)}...")
    signals: SignalCollection = await pipeline.extract(
        tickers=WATCHLIST,
        filing_types=["10-K"],
        lookback_years=2,
        engines=["risk_differ"],
    )

    # All risk changes
    risk_signals = signals.by_type("risk_change")
    print(f"\nTotal risk-factor changes: {len(risk_signals)}")

    # Critical and high severity
    critical = signals.risk_changes(severity="CRITICAL")
    high = signals.risk_changes(severity="HIGH")

    if len(critical) > 0:
        print(f"\n{'=' * 60}")
        print(f"  CRITICAL RISK CHANGES ({len(critical)})")
        print(f"{'=' * 60}")
        for sig in critical:
            print(f"  [{sig.ticker}] {sig.context}")
            if sig.related_tickers:
                print(f"    Related: {', '.join(sig.related_tickers)}")

    if len(high) > 0:
        print(f"\n{'=' * 60}")
        print(f"  HIGH SEVERITY CHANGES ({len(high)})")
        print(f"{'=' * 60}")
        for sig in high:
            print(f"  [{sig.ticker}] {sig.context}")

    # Per-company summary
    print(f"\n{'=' * 60}")
    print("  PER-COMPANY SUMMARY")
    print(f"{'=' * 60}")
    for ticker in WATCHLIST:
        ticker_risks = risk_signals.by_ticker(ticker)
        bearish = ticker_risks.by_direction("bearish")
        bullish = ticker_risks.by_direction("bullish")
        print(
            f"  {ticker}: {len(ticker_risks)} changes "
            f"({len(bearish)} bearish, {len(bullish)} bullish)"
        )

    # Export
    signals.to_parquet("risk_monitor.parquet")
    print("\nExported to risk_monitor.parquet")


if __name__ == "__main__":
    asyncio.run(main())
