"""Management tone analysis engine.

Goes beyond simple positive/negative polarity to track **topic-level
tone trajectories** across consecutive filings.  For each topic (e.g.
"revenue growth", "AI spending", "margins"), the engine classifies
management's tone along a six-point scale and detects directional
shifts.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import structlog

from sigint.engines.base import BaseEngine
from sigint.llm import LLMClient
from sigint.models import (
    FilingSection,
    Signal,
    SignalDirection,
    SignalType,
    ToneLabel,
)

logger = structlog.get_logger()

_SYSTEM_PROMPT = """\
You are a qualitative research analyst specialising in management
communication patterns.  Given the MD&A (Management Discussion and
Analysis) section of an SEC filing, identify the top business topics
discussed and classify management's tone on each topic.

Tone labels (from most bullish to most bearish):
1. confident_expanding -- Strong conviction, expansion language
2. optimistic_cautious -- Positive but measured
3. neutral_factual -- Purely factual, no directional language
4. hedging_cautious -- Qualifiers, hedging language ("may", "could")
5. defensive_justifying -- Defending past decisions, explaining misses
6. pessimistic_warning -- Explicit warnings, negative outlook

For each topic, return a JSON object:
- topic: Short topic label (e.g. "revenue growth", "AI spending")
- tone: One of the six tone labels above
- confidence: 0.0 to 1.0
- key_phrases: 2-3 short quotes that justify your classification

Return a JSON array of topics (aim for 3-8 topics per filing).
If the section is too short or uninformative, return [].
"""

_TONE_RANK: dict[ToneLabel, int] = {
    ToneLabel.CONFIDENT_EXPANDING: 5,
    ToneLabel.OPTIMISTIC_CAUTIOUS: 4,
    ToneLabel.NEUTRAL_FACTUAL: 3,
    ToneLabel.HEDGING_CAUTIOUS: 2,
    ToneLabel.DEFENSIVE_JUSTIFYING: 1,
    ToneLabel.PESSIMISTIC_WARNING: 0,
}

# Direction and strength for a tone in isolation (no prior filing to compare)
_TONE_BASELINE: dict[ToneLabel, tuple[SignalDirection, float]] = {
    ToneLabel.CONFIDENT_EXPANDING: (SignalDirection.BULLISH, 0.70),
    ToneLabel.OPTIMISTIC_CAUTIOUS: (SignalDirection.BULLISH, 0.40),
    ToneLabel.NEUTRAL_FACTUAL: (SignalDirection.NEUTRAL, 0.0),
    ToneLabel.HEDGING_CAUTIOUS: (SignalDirection.BEARISH, 0.30),
    ToneLabel.DEFENSIVE_JUSTIFYING: (SignalDirection.BEARISH, 0.55),
    ToneLabel.PESSIMISTIC_WARNING: (SignalDirection.BEARISH, 0.80),
}


def classify_tone_shift(
    current: ToneLabel, previous: ToneLabel
) -> tuple[SignalDirection, float]:
    """Determine direction and strength of a tone shift.

    Args:
        current: Tone in the current filing.
        previous: Tone in the prior filing.

    Returns:
        Tuple of (direction, strength).  Strength is proportional to
        the magnitude of the shift on the six-point scale.
    """
    delta = _TONE_RANK[current] - _TONE_RANK[previous]
    if delta == 0:
        return SignalDirection.NEUTRAL, 0.0
    direction = SignalDirection.BULLISH if delta > 0 else SignalDirection.BEARISH
    # Normalise: max possible shift is 5 steps
    strength = min(1.0, abs(delta) / 5.0)
    return direction, strength


class ToneEngine(BaseEngine):
    """Track topic-level management tone across filings."""

    @property
    def name(self) -> str:
        return "tone"

    async def extract(
        self,
        sections: Sequence[FilingSection],
        llm: LLMClient,
        *,
        previous_sections: Sequence[FilingSection] | None = None,
    ) -> list[Signal]:
        """Analyse management tone in MD&A and compare to prior filing.

        Args:
            sections: Parsed sections from the current filing.
            llm: LLM client for extraction.
            previous_sections: Parsed sections from the prior filing.

        Returns:
            Signals for each topic where a tone shift was detected.
        """
        current_mda = _find_mda(sections)
        if current_mda is None:
            logger.info(
                "tone_no_mda",
                ticker=sections[0].ticker if sections else "?",
            )
            return []

        current_tones = await _extract_tones(current_mda, llm)
        if not current_tones:
            return []

        # If we have a previous filing, detect shifts
        if previous_sections:
            previous_mda = _find_mda(previous_sections)
            if previous_mda:
                previous_tones = await _extract_tones(previous_mda, llm)
                return _compute_shifts(
                    current_tones,
                    previous_tones,
                    current_mda,
                    previous_mda,
                )

        # No previous filing available — emit baseline tone signals so the
        # first filing in a series is not silently dropped.
        logger.info(
            "tone_baseline_only",
            ticker=current_mda.ticker,
            filing=current_mda.filed_date.isoformat(),
        )
        return _compute_baselines(current_tones, current_mda)


def _find_mda(
    sections: Sequence[FilingSection],
) -> FilingSection | None:
    for s in sections:
        if s.section_key == "md_and_a":
            return s
    return None


async def _extract_tones(
    section: FilingSection, llm: LLMClient
) -> list[dict[str, Any]]:
    """Call the LLM to extract per-topic tone from an MD&A section."""
    text = section.text[:50_000]
    user_msg = (
        f"Company: {section.ticker}\n"
        f"Filing: {section.filing_type.value} "
        f"filed {section.filed_date.isoformat()}\n\n"
        f"--- BEGIN MD&A ---\n{text}\n--- END MD&A ---"
    )
    raw_items = await llm.extract_json(_SYSTEM_PROMPT, user_msg, temperature=0.0)
    if not isinstance(raw_items, list):
        raw_items = [raw_items]
    return raw_items


def _compute_baselines(
    current_tones: list[dict[str, Any]],
    current_section: FilingSection,
) -> list[Signal]:
    """Emit tone signals for a first filing with no prior comparison.

    Uses the absolute tone position (rather than a shift) to assign
    direction and strength, so the first filing in a series is not lost.
    """
    signals: list[Signal] = []
    for item in current_tones:
        topic = str(item.get("topic", "")).lower().strip()
        if not topic:
            continue
        try:
            current_tone = ToneLabel(item.get("tone", "neutral_factual"))
        except ValueError:
            continue

        direction, strength = _TONE_BASELINE[current_tone]
        if direction == SignalDirection.NEUTRAL:
            continue

        confidence = float(item.get("confidence", 0.5))
        signals.append(
            Signal(
                timestamp=datetime.combine(
                    current_section.filed_date,
                    datetime.min.time(),
                    tzinfo=UTC,
                ),
                ticker=current_section.ticker,
                signal_type=SignalType.TONE_SHIFT,
                direction=direction,
                strength=strength,
                confidence=confidence,
                context=(
                    f"Baseline tone on '{topic}': {current_tone.value} "
                    f"(no prior filing available)"
                ),
                source_filing="",
                related_tickers=[],
                metadata={
                    "topic": topic,
                    "current_tone": current_tone.value,
                    "previous_tone": None,
                    "key_phrases": item.get("key_phrases", []),
                    "current_filing": (
                        f"{current_section.filing_type.value} "
                        f"{current_section.filed_date.isoformat()}"
                    ),
                    "previous_filing": None,
                    "is_baseline": True,
                },
            )
        )
    return signals


def _compute_shifts(
    current_tones: list[dict[str, Any]],
    previous_tones: list[dict[str, Any]],
    current_section: FilingSection,
    previous_section: FilingSection,
) -> list[Signal]:
    """Compare tone on matching topics across filings."""
    prev_by_topic: dict[str, dict[str, Any]] = {}
    for t in previous_tones:
        topic = str(t.get("topic", "")).lower().strip()
        if topic:
            prev_by_topic[topic] = t

    signals: list[Signal] = []
    for item in current_tones:
        topic = str(item.get("topic", "")).lower().strip()
        if not topic:
            continue

        try:
            current_tone = ToneLabel(item.get("tone", "neutral_factual"))
        except ValueError:
            continue

        confidence = float(item.get("confidence", 0.5))

        # Find the closest matching previous topic
        prev = prev_by_topic.get(topic)
        if prev is None:
            continue

        try:
            previous_tone = ToneLabel(prev.get("tone", "neutral_factual"))
        except ValueError:
            continue

        direction, strength = classify_tone_shift(current_tone, previous_tone)
        if direction == SignalDirection.NEUTRAL:
            continue

        signals.append(
            Signal(
                timestamp=datetime.combine(
                    current_section.filed_date,
                    datetime.min.time(),
                    tzinfo=UTC,
                ),
                ticker=current_section.ticker,
                signal_type=SignalType.TONE_SHIFT,
                direction=direction,
                strength=strength,
                confidence=confidence,
                context=(
                    f"Tone on '{topic}' shifted from "
                    f"{previous_tone.value} to {current_tone.value}"
                ),
                source_filing="",
                related_tickers=[],
                metadata={
                    "topic": topic,
                    "current_tone": current_tone.value,
                    "previous_tone": previous_tone.value,
                    "key_phrases": item.get("key_phrases", []),
                    "current_filing": (
                        f"{current_section.filing_type.value} "
                        f"{current_section.filed_date.isoformat()}"
                    ),
                    "previous_filing": (
                        f"{previous_section.filing_type.value} "
                        f"{previous_section.filed_date.isoformat()}"
                    ),
                },
            )
        )

    return signals
