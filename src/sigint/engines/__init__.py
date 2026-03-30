"""Extraction engines for sigint.

Each engine implements a common async interface::

    async def extract(sections, llm, **kwargs) -> list[Signal]

Engines:
    supply_chain -- Build a supplier/customer knowledge graph.
    risk_differ  -- Diff risk factors between consecutive filings.
    m_and_a      -- Detect M&A language patterns.
    tone         -- Track topic-level management tone trajectories.
"""

from sigint.engines.m_and_a import MandAEngine
from sigint.engines.risk_differ import RiskDifferEngine
from sigint.engines.supply_chain import SupplyChainEngine
from sigint.engines.tone import ToneEngine

__all__ = [
    "MandAEngine",
    "RiskDifferEngine",
    "SupplyChainEngine",
    "ToneEngine",
]
