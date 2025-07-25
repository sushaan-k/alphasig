#!/usr/bin/env python3
"""Offline demo for sigint."""

from __future__ import annotations

from datetime import UTC, datetime

from sigint import Signal, SignalCollection, SignalDirection, SignalType


def build_signal(
    ticker: str,
    signal_type: SignalType,
    direction: SignalDirection,
    strength: float,
    context: str,
    metadata: dict[str, object],
) -> Signal:
    return Signal(
        timestamp=datetime.now(UTC),
        ticker=ticker,
        signal_type=signal_type,
        direction=direction,
        strength=strength,
        confidence=0.84,
        context=context,
        source_filing="https://www.sec.gov/Archives/demo",
        metadata=metadata,
    )


def main() -> None:
    signals = SignalCollection(
        [
            build_signal(
                "AAPL",
                SignalType.SUPPLY_CHAIN,
                SignalDirection.BEARISH,
                0.76,
                "Apple disclosed higher single-supplier concentration.",
                {
                    "target": "TSMC",
                    "relation": "depends_on",
                    "edge_context": "advanced node fabrication capacity",
                },
            ),
            build_signal(
                "NVDA",
                SignalType.RISK_CHANGE,
                SignalDirection.BULLISH,
                0.71,
                "Management removed language around acute inventory pressure.",
                {"severity": "MEDIUM"},
            ),
        ]
    )

    graph = signals.supply_chain_graph()
    bearish = signals.by_direction("bearish").above_strength(0.7)

    print("sigint demo")
    print(f"total signals: {len(signals)}")
    print(f"bearish signals: {len(bearish)}")
    print(f"supply-chain nodes: {len(graph.nodes)}")
    print(f"supply-chain edges: {graph.edge_count}")


if __name__ == "__main__":
    main()
