"""M&A signal detection engine.

Scans filings for language patterns that historically precede mergers,
acquisitions, divestitures, and other strategic transactions.  The
detection is based on categories derived from academic M&A research:

* **Strategic alternatives**: "exploring strategic alternatives",
  "potential transactions", "retained advisors".
* **Advisor engagement**: New mentions of investment banks or legal
  counsel in transaction-related contexts.
* **Cash positioning**: Unusual commentary on cash reserves, credit
  facilities, or financing capacity.
* **Board changes**: New directors with M&A or PE backgrounds.
* **Related-party transactions**: Unusual related-party disclosures
  that may signal insider knowledge of a pending deal.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import structlog

from sigint.engines.base import BaseEngine
from sigint.llm import LLMClient
from sigint.models import (
    FilingSection,
    MandAIndicator,
    Signal,
    SignalDirection,
    SignalType,
)

logger = structlog.get_logger()

_SYSTEM_PROMPT = """\
You are an M&A (mergers and acquisitions) analyst reviewing an SEC
filing.  Identify any language that may signal upcoming M&A activity,
including but not limited to:

1. **Strategic alternatives** -- phrases like "exploring strategic
   alternatives", "evaluating potential transactions", "engaged
   financial advisors", "sale of all or part".
2. **Advisor engagement** -- new mentions of investment banks,
   financial advisors, or special counsel in a deal context.
3. **Cash positioning** -- unusual emphasis on cash reserves, new
   credit facilities, or bridge financing that could fund an
   acquisition or signal preparedness.
4. **Board changes** -- new directors whose backgrounds suggest M&A
   expertise (PE, investment banking, corporate development).
5. **Related-party transactions** -- unusual or new related-party
   disclosures that could indicate insider deal activity.

For each indicator, return a JSON object:
- indicator: A short label for the signal (e.g. "strategic alternatives
  language")
- category: one of "strategic_alternatives", "advisor_engagement",
  "cash_positioning", "board_change", "related_party"
- excerpt: The exact excerpt from the filing (keep it short, 1-3
  sentences)
- confidence: 0.0 to 1.0

Return a JSON array.  If no M&A indicators are found, return [].
Be selective -- only flag language that would make an experienced M&A
banker take notice.
"""

# Categories ranked by predictive strength (from M&A research)
_CATEGORY_WEIGHT: dict[str, float] = {
    "strategic_alternatives": 0.95,
    "advisor_engagement": 0.80,
    "cash_positioning": 0.60,
    "board_change": 0.55,
    "related_party": 0.70,
}


class MandAEngine(BaseEngine):
    """Detect M&A language patterns in SEC filings."""

    @property
    def name(self) -> str:
        return "m_and_a"

    async def extract(
        self,
        sections: Sequence[FilingSection],
        llm: LLMClient,
        *,
        previous_sections: Sequence[FilingSection] | None = None,
    ) -> list[Signal]:
        """Scan filing sections for M&A indicators.

        Args:
            sections: Parsed sections from the current filing.
            llm: LLM client for extraction.
            previous_sections: Unused by this engine.

        Returns:
            Signals for detected M&A language.
        """
        # M&A signals can appear in almost any section
        if not sections:
            return []

        indicators: list[MandAIndicator] = []
        ticker = sections[0].ticker
        filing_type = sections[0].filing_type
        filed_date = sections[0].filed_date

        for section in sections:
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
                    ind = MandAIndicator(
                        ticker=ticker,
                        indicator=str(item.get("indicator", "")),
                        category=str(item.get("category", "strategic_alternatives")),
                        excerpt=str(item.get("excerpt", "")),
                        confidence=float(item.get("confidence", 0.5)),
                        filing_type=filing_type,
                        filed_date=filed_date,
                    )
                    indicators.append(ind)
                except (ValueError, KeyError) as exc:
                    logger.warning(
                        "m_and_a_parse_error",
                        error=str(exc),
                        item=str(item)[:200],
                    )

        signals = _indicators_to_signals(indicators)
        logger.info(
            "m_and_a_extracted",
            ticker=ticker,
            indicators=len(indicators),
            signals=len(signals),
        )
        return signals


def _indicators_to_signals(
    indicators: list[MandAIndicator],
) -> list[Signal]:
    """Convert M&A indicators into standardised Signal objects."""
    if not indicators:
        return []

    # Aggregate: if multiple indicators, boost overall signal strength
    aggregate_confidence = min(
        1.0,
        sum(i.confidence for i in indicators) / len(indicators),
    )

    # Compute a composite strength based on category weights
    weighted_sum = sum(
        _CATEGORY_WEIGHT.get(i.category, 0.5) * i.confidence for i in indicators
    )
    composite_strength = min(1.0, weighted_sum / max(len(indicators), 1))

    # Infer direction from the categories present:
    # - strategic_alternatives / advisor_engagement / board_change strongly
    #   suggest the company is a *target* expecting an acquisition premium → BULLISH
    # - cash_positioning alone (large cash build without deal language) often
    #   signals the company may be an *acquirer* (cash outflow risk) → BEARISH
    # - related_party alone is ambiguous → NEUTRAL
    target_categories = {"strategic_alternatives", "advisor_engagement", "board_change"}
    acquirer_categories = {"cash_positioning"}

    has_target_signal = any(
        i.category in target_categories and i.confidence > 0.6 for i in indicators
    )
    has_strong_strategic = any(
        i.category == "strategic_alternatives" and i.confidence > 0.7
        for i in indicators
    )
    target_only = all(
        i.category in acquirer_categories | {"related_party"} for i in indicators
    )

    if has_target_signal or has_strong_strategic:
        direction = SignalDirection.BULLISH  # Target premium expected
    elif target_only and any(i.category == "cash_positioning" for i in indicators):
        direction = SignalDirection.BEARISH  # Likely acquirer — capital outflow risk
    else:
        direction = SignalDirection.NEUTRAL

    ticker = indicators[0].ticker
    filed_date = indicators[0].filed_date

    # One summary signal
    summary_parts = [
        f"{i.category}: {i.indicator} (conf={i.confidence:.2f})" for i in indicators[:5]
    ]
    context = "; ".join(summary_parts)

    signals: list[Signal] = [
        Signal(
            timestamp=datetime.combine(filed_date, datetime.min.time(), tzinfo=UTC),
            ticker=ticker,
            signal_type=SignalType.M_AND_A,
            direction=direction,
            strength=composite_strength,
            confidence=aggregate_confidence,
            context=f"M&A indicators detected: {context}",
            source_filing="",
            related_tickers=[],
            metadata={
                "indicator_count": len(indicators),
                "categories": list({i.category for i in indicators}),
                "indicators": [
                    {
                        "indicator": i.indicator,
                        "category": i.category,
                        "confidence": i.confidence,
                        "excerpt": i.excerpt[:200],
                    }
                    for i in indicators
                ],
            },
        )
    ]

    return signals
