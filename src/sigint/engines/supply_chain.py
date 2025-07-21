"""Supply-chain graph extraction engine.

Reads the Business and Risk Factors sections of 10-K / 10-Q filings and
uses the LLM to identify supplier, customer, and partner relationships.
The output is a list of :class:`SupplyChainEdge` instances and
corresponding :class:`Signal` objects.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import structlog

from sigint.engines.base import BaseEngine
from sigint.llm import LLMClient
from sigint.models import (
    FilingSection,
    RelationType,
    Signal,
    SignalDirection,
    SignalType,
    SupplyChainEdge,
)

logger = structlog.get_logger()

_SYSTEM_PROMPT = """\
You are a financial analyst specialising in supply-chain analysis of
SEC filings.  Given the text of a filing section, extract every
supplier, customer, manufacturing partner, or critical dependency
mentioned.

For each relationship return a JSON object with these fields:
- source: the ticker of the company that filed (provided to you)
- target: the name of the other company (use the ticker if you know it,
  otherwise use the company name as written in the filing)
- relation: one of "depends_on", "supplies_to", "partners_with"
- context: a short phrase describing what the relationship is about
  (e.g. "semiconductor manufacturing", "cloud hosting")
- confidence: your confidence in the extraction, 0.0 to 1.0

Return a JSON array.  If you find no relationships, return [].
"""


class SupplyChainEngine(BaseEngine):
    """Extract supplier/customer/partner edges from filing text."""

    @property
    def name(self) -> str:
        return "supply_chain"

    async def extract(
        self,
        sections: Sequence[FilingSection],
        llm: LLMClient,
        *,
        previous_sections: Sequence[FilingSection] | None = None,
    ) -> list[Signal]:
        """Extract supply-chain signals from Business and Risk sections.

        Args:
            sections: Parsed sections from a single filing.
            llm: LLM client for extraction.
            previous_sections: Unused by this engine.

        Returns:
            One :class:`Signal` per detected supply-chain edge.
        """
        relevant_keys = {"business", "risk_factors", "md_and_a"}
        target_sections = [s for s in sections if s.section_key in relevant_keys]
        if not target_sections:
            logger.info(
                "supply_chain_no_sections",
                ticker=sections[0].ticker if sections else "?",
            )
            return []

        edges: list[SupplyChainEdge] = []
        for section in target_sections:
            # Truncate very long sections to stay within context limits
            text = section.text[:50_000]
            user_msg = (
                f"Company ticker: {section.ticker}\n"
                f"Filing type: {section.filing_type.value}\n"
                f"Section: {section.section_name}\n\n"
                f"--- BEGIN FILING TEXT ---\n{text}\n--- END FILING TEXT ---"
            )

            raw_items = await llm.extract_json(
                _SYSTEM_PROMPT, user_msg, temperature=0.0
            )
            if not isinstance(raw_items, list):
                raw_items = [raw_items]

            for item in raw_items:
                try:
                    edge = SupplyChainEdge(
                        source=section.ticker,
                        target=str(item.get("target", "")),
                        relation=RelationType(item.get("relation", "depends_on")),
                        context=str(item.get("context", "")),
                        confidence=float(item.get("confidence", 0.5)),
                        filing_type=section.filing_type,
                        filed_date=section.filed_date,
                    )
                    edges.append(edge)
                except (ValueError, KeyError) as exc:
                    logger.warning(
                        "supply_chain_edge_parse_error",
                        error=str(exc),
                        item=str(item)[:200],
                    )

        # Deduplicate edges by (source, target, relation)
        seen: set[tuple[str, str, str]] = set()
        unique_edges: list[SupplyChainEdge] = []
        for e in edges:
            key = (e.source, e.target, e.relation.value)
            if key not in seen:
                seen.add(key)
                unique_edges.append(e)

        signals = _edges_to_signals(unique_edges)
        logger.info(
            "supply_chain_extracted",
            ticker=sections[0].ticker if sections else "?",
            edges=len(unique_edges),
            signals=len(signals),
        )
        return signals


def _edges_to_signals(edges: list[SupplyChainEdge]) -> list[Signal]:
    """Convert supply-chain edges into standardised Signal objects."""
    signals: list[Signal] = []
    for edge in edges:
        signals.append(
            Signal(
                timestamp=datetime.combine(
                    edge.filed_date,
                    datetime.min.time(),
                    tzinfo=UTC,
                ),
                ticker=edge.source,
                signal_type=SignalType.SUPPLY_CHAIN,
                direction=SignalDirection.NEUTRAL,
                strength=edge.confidence,
                confidence=edge.confidence,
                context=(
                    f"{edge.source} {edge.relation.value} {edge.target} "
                    f"({edge.context})"
                ),
                source_filing="",  # Filled by pipeline
                related_tickers=[edge.target],
                metadata={
                    "target": edge.target,
                    "relation": edge.relation.value,
                    "edge_context": edge.context,
                    "filing_type": edge.filing_type.value,
                },
            )
        )
    return signals
