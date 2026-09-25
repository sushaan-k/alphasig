"""Supply-chain graph extraction engine.

Reads the Business and Risk Factors sections of 10-K / 10-Q filings and
uses the LLM to identify supplier, customer, and partner relationships.
The output is a list of :class:`SupplyChainEdge` instances and
corresponding :class:`Signal` objects.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import structlog

from alphasig.engines.base import BaseEngine, json_objects
from alphasig.llm import LLMClient
from alphasig.models import (
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
- target: the other company -- its upper-case stock ticker if it is
  publicly traded and you know it (e.g. "TSM"), otherwise its name as
  written in the filing
- relation: one of "depends_on", "supplies_to", "partners_with"
- context: a short phrase describing what the relationship is about
  (e.g. "semiconductor manufacturing", "cloud hosting")
- confidence: your confidence in the extraction, 0.0 to 1.0
- exposure: only if the filing states a concentration figure for this
  relationship (e.g. "accounted for 22% of net sales"), that share as a
  fraction between 0 and 1 (0.22); otherwise null.  Never estimate it.

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

        # Sections are independent; the LLM client bounds concurrency.
        responses = await asyncio.gather(
            *(_scan_section(section, llm) for section in target_sections)
        )
        edges: list[SupplyChainEdge] = []
        for section, raw_items in zip(target_sections, responses, strict=True):
            for item in json_objects(raw_items):
                target = str(item.get("target") or "").strip()
                if not target or target.upper() == section.ticker.upper():
                    continue
                try:
                    edge = SupplyChainEdge(
                        source=section.ticker,
                        target=target,
                        relation=RelationType(item.get("relation", "depends_on")),
                        context=str(item.get("context", "")),
                        confidence=float(item.get("confidence", 0.5)),
                        exposure=_parse_exposure(item.get("exposure")),
                        filing_type=section.filing_type,
                        filed_date=section.filed_date,
                    )
                    edges.append(edge)
                except (ValueError, KeyError, TypeError) as exc:
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

        signals = _edges_to_signals(unique_edges, target_sections[0])
        logger.info(
            "supply_chain_extracted",
            ticker=sections[0].ticker if sections else "?",
            edges=len(unique_edges),
            signals=len(signals),
        )
        return signals


def _parse_exposure(value: Any) -> float | None:
    """Coerce a stated concentration share to a 0-1 fraction, else ``None``.

    Values in (1, 100] are read as percentages ("22" -> 0.22); anything
    unparseable or out of range is dropped rather than discarding the edge.
    """
    try:
        share = float(value)
    except (TypeError, ValueError):
        return None
    if 1.0 < share <= 100.0:
        share /= 100.0
    return share if 0.0 <= share <= 1.0 else None


async def _scan_section(section: FilingSection, llm: LLMClient) -> Any:
    # Truncate very long sections to stay within context limits
    text = section.text[:50_000]
    user_msg = (
        f"Company ticker: {section.ticker}\n"
        f"Filing type: {section.filing_type.value}\n"
        f"Section: {section.section_name}\n\n"
        f"--- BEGIN FILING TEXT ---\n{text}\n--- END FILING TEXT ---"
    )
    return await llm.extract_json(_SYSTEM_PROMPT, user_msg)


def _edges_to_signals(
    edges: list[SupplyChainEdge], section: FilingSection
) -> list[Signal]:
    """Convert supply-chain edges into standardised Signal objects."""
    signals: list[Signal] = []
    for edge in edges:
        signals.append(
            Signal(
                timestamp=section.available_at,
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
                    "exposure": edge.exposure,
                    "filing_type": edge.filing_type.value,
                    "filed_date": edge.filed_date.isoformat(),
                },
            )
        )
    return signals
