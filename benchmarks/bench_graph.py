"""Supply-chain graph construction scaling.

Edges are synthetic: a preferential-attachment-like supplier distribution
(a few hub suppliers such as foundries, a long tail of small ones), with
~10% duplicate (source, target, relation) edges so the confidence-merge path
in :meth:`SupplyChainGraph.add_edges` is exercised.
"""

from __future__ import annotations

import random
import time
from datetime import UTC, date, datetime
from typing import Any

from alphasig.graph import SupplyChainGraph
from alphasig.models import (
    FilingType,
    RelationType,
    Signal,
    SignalDirection,
    SignalType,
    SupplyChainEdge,
)
from alphasig.signals import SignalCollection
from benchmarks._common import SEED

_REL = list(RelationType)


def make_edges(n: int, seed: int = SEED) -> list[SupplyChainEdge]:
    rng = random.Random(seed)
    n_companies = max(50, n // 10)
    edges = []
    for _ in range(n):
        src = f"C{rng.randrange(n_companies)}"
        tgt = f"S{int(rng.paretovariate(1.2)) % n_companies}"
        edges.append(
            SupplyChainEdge(
                source=src,
                target=tgt,
                relation=_REL[rng.randrange(3)],
                context="component supply",
                confidence=round(rng.random(), 3),
                filing_type=FilingType.TEN_K,
                filed_date=date(2024, 1, 1),
            )
        )
    return edges


def _signals_from_edges(edges: list[SupplyChainEdge]) -> list[Signal]:
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    return [
        Signal(
            timestamp=ts,
            ticker=e.source,
            signal_type=SignalType.SUPPLY_CHAIN,
            direction=SignalDirection.NEUTRAL,
            strength=e.confidence,
            confidence=e.confidence,
            context="",
            source_filing="",
            related_tickers=[e.target],
            metadata={
                "target": e.target,
                "relation": e.relation.value,
                "edge_context": e.context,
                "filing_type": "10-K",
            },
        )
        for e in edges
    ]


def run(quick: bool = False) -> dict[str, Any]:
    sizes = (1_000, 10_000) if quick else (1_000, 10_000, 100_000)
    rows = []
    for n in sizes:
        edges = make_edges(n)
        t0 = time.perf_counter()
        g = SupplyChainGraph(edges)
        build = time.perf_counter() - t0

        hub = g.most_connected(1)[0][0]
        t0 = time.perf_counter()
        exp = g.exposure(hub)
        exposure_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        g.most_connected(10)
        top_s = time.perf_counter() - t0

        coll = SignalCollection(_signals_from_edges(edges))
        t0 = time.perf_counter()
        coll.supply_chain_graph()
        from_signals = time.perf_counter() - t0

        rows.append(
            {
                "edges_in": n,
                "nodes": len(g.nodes),
                "edges_kept": g.edge_count,
                "build_s": round(build, 4),
                "edges_per_s": round(n / build),
                "exposure_hub_ms": round(exposure_s * 1000, 3),
                "hub_total_exposed": exp["total_exposed"],
                "most_connected_ms": round(top_s * 1000, 3),
                "from_signal_collection_s": round(from_signals, 4),
            }
        )
    return {"sizes": rows}
