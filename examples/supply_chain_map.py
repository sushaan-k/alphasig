"""Build and visualise a supply-chain dependency graph.

Extracts supplier/customer relationships from 10-K filings and builds
a NetworkX graph.  Shows which companies are most exposed to disruptions
at key suppliers like TSMC.

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    export ALPHASIG_USER_AGENT="Your Name you@example.com"
    python examples/supply_chain_map.py
"""

from __future__ import annotations

import asyncio

from alphasig import Pipeline

TICKERS = ["AAPL", "NVDA", "AMD", "QCOM", "AVGO", "INTC"]


async def main() -> None:
    pipeline = Pipeline(
        # Model and EDGAR User-Agent come from ALPHASIG_MODEL (optional) and
        # ALPHASIG_USER_AGENT ("Your Name you@example.com", required by SEC).
        cache_dir="./edgar_cache",
        db_path=None,  # Don't persist for this example
    )

    print(f"Extracting supply chain from {', '.join(TICKERS)}...")
    signals = await pipeline.extract(
        tickers=TICKERS,
        filing_types=["10-K"],
        lookback_years=1,
        engines=["supply_chain"],
        store=False,
    )

    # Build the graph
    graph = signals.supply_chain_graph()
    print(f"\nGraph: {len(graph.nodes)} nodes, {graph.edge_count} edges")

    # Who depends on TSMC?  Public counterparties are keyed by ticker (TSM).
    tsmc_exposure = graph.exposure("TSM")
    print("\nTSMC (TSM) exposure analysis:")
    print(f"  Direct dependents: {tsmc_exposure['direct_dependents']}")
    print(f"  Transitive dependents: {tsmc_exposure['transitive_dependents']}")
    print(f"  Exposure score: {tsmc_exposure['exposure_score']:.2%}")

    # Most connected entities
    print("\nMost connected entities:")
    for name, degree in graph.most_connected(5):
        print(f"  {name}: {degree} connections")

    # Check for paths
    for source, target in [("AAPL", "TSM"), ("NVDA", "TSM")]:
        path = graph.shortest_path(source, target)
        if path:
            print(f"\n  Path {source} -> {target}: {' -> '.join(path)}")

    # Save visualisation
    try:
        graph.plot(output_path="supply_chain_graph.png")
        print("\nGraph saved to supply_chain_graph.png")
    except ImportError:
        print("\nInstall matplotlib to generate the graph visualisation")


if __name__ == "__main__":
    asyncio.run(main())
