"""Risk-factor diffing engine.

Compares the Risk Factors section (Item 1A) between consecutive
10-K / 10-Q filings for the same company.  Classifies each change as
NEW, REMOVED, ESCALATED, or DE_ESCALATED and estimates severity.

Inspired by the "Lazy Prices" paper (Cohen, Malloy, Nguyen 2020) which
demonstrated that 10-K language changes strongly predict future returns.
"""

from __future__ import annotations

import difflib
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import structlog

from sigint.engines.base import BaseEngine
from sigint.llm import LLMClient
from sigint.models import (
    FilingSection,
    RiskChange,
    RiskChangeType,
    Severity,
    Signal,
    SignalDirection,
    SignalType,
)

logger = structlog.get_logger()

_SYSTEM_PROMPT = """\
You are a securities lawyer and risk analyst.  You will be given two
versions of the Risk Factors section from consecutive SEC filings by
the same company.

Perform a detailed diff and classify each material change:
- NEW: A risk factor that did not appear in the previous filing.
- REMOVED: A risk factor present before but absent now.
- ESCALATED: A risk factor whose language became more severe
  (e.g. "may" became "is", "could" became "currently").
- DE_ESCALATED: A risk factor whose language became less severe.

For each change, return a JSON object with:
- change_type: "NEW" | "REMOVED" | "ESCALATED" | "DE_ESCALATED"
- risk: A concise description of the risk factor (1-2 sentences)
- language_shift: For ESCALATED/DE_ESCALATED, quote the old and new
  phrasing.  For NEW/REMOVED, leave empty.
- severity_estimate: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
- related_tickers: list of tickers of other companies mentioned in
  the risk factor (if any).  Empty list if none.

Return a JSON array.  If there are no material changes, return [].
Focus on substantive, investable changes -- not boilerplate rewording.
"""

_DIRECTION_MAP: dict[RiskChangeType, SignalDirection] = {
    RiskChangeType.NEW: SignalDirection.BEARISH,
    RiskChangeType.REMOVED: SignalDirection.BULLISH,
    RiskChangeType.ESCALATED: SignalDirection.BEARISH,
    RiskChangeType.DE_ESCALATED: SignalDirection.BULLISH,
}

_SEVERITY_STRENGTH: dict[Severity, float] = {
    Severity.LOW: 0.25,
    Severity.MEDIUM: 0.50,
    Severity.HIGH: 0.75,
    Severity.CRITICAL: 0.95,
}


def compute_text_similarity(text_a: str, text_b: str) -> float:
    """Compute a 0-1 similarity ratio between two text blocks.

    Uses :class:`difflib.SequenceMatcher` on whitespace-normalised
    text.  A ratio below ~0.85 usually indicates substantive changes.
    """
    a_norm = " ".join(text_a.split())
    b_norm = " ".join(text_b.split())
    return difflib.SequenceMatcher(None, a_norm, b_norm).ratio()


class RiskDifferEngine(BaseEngine):
    """Diff risk factors between consecutive filings."""

    @property
    def name(self) -> str:
        return "risk_differ"

    async def extract(
        self,
        sections: Sequence[FilingSection],
        llm: LLMClient,
        *,
        previous_sections: Sequence[FilingSection] | None = None,
    ) -> list[Signal]:
        """Compare current and previous Risk Factors sections.

        Args:
            sections: Parsed sections from the current filing.
            llm: LLM client for structured extraction.
            previous_sections: Parsed sections from the prior filing
                of the same type for the same company.

        Returns:
            Signals representing material risk-factor changes.
        """
        current_rf = _find_risk_factors(sections)
        if current_rf is None:
            logger.info(
                "risk_differ_no_current_rf",
                ticker=sections[0].ticker if sections else "?",
            )
            return []

        if not previous_sections:
            logger.info(
                "risk_differ_no_previous",
                ticker=current_rf.ticker,
            )
            return []

        previous_rf = _find_risk_factors(previous_sections)
        if previous_rf is None:
            logger.info(
                "risk_differ_no_previous_rf",
                ticker=current_rf.ticker,
            )
            return []

        # Quick similarity check -- skip LLM call if nearly identical
        similarity = compute_text_similarity(current_rf.text, previous_rf.text)
        if similarity > 0.98:
            logger.info(
                "risk_differ_no_material_change",
                ticker=current_rf.ticker,
                similarity=round(similarity, 4),
            )
            return []

        # Truncate to fit context window
        current_text = current_rf.text[:40_000]
        previous_text = previous_rf.text[:40_000]

        user_msg = (
            f"Company: {current_rf.ticker}\n"
            f"Current filing: {current_rf.filing_type.value} "
            f"filed {current_rf.filed_date.isoformat()}\n"
            f"Previous filing: {previous_rf.filing_type.value} "
            f"filed {previous_rf.filed_date.isoformat()}\n\n"
            f"--- PREVIOUS RISK FACTORS ---\n{previous_text}\n\n"
            f"--- CURRENT RISK FACTORS ---\n{current_text}\n"
        )

        raw_items = await llm.extract_json(_SYSTEM_PROMPT, user_msg, temperature=0.0)
        if not isinstance(raw_items, list):
            raw_items = [raw_items]

        changes: list[RiskChange] = []
        for item in raw_items:
            try:
                change = _parse_risk_change(item, current_rf, previous_rf)
                changes.append(change)
            except (ValueError, KeyError) as exc:
                logger.warning(
                    "risk_differ_parse_error",
                    error=str(exc),
                    item=str(item)[:200],
                )

        signals = _changes_to_signals(changes, current_rf)
        logger.info(
            "risk_differ_extracted",
            ticker=current_rf.ticker,
            changes=len(changes),
            signals=len(signals),
            similarity=round(similarity, 4),
        )
        return signals


def _find_risk_factors(
    sections: Sequence[FilingSection],
) -> FilingSection | None:
    for s in sections:
        if s.section_key == "risk_factors":
            return s
    return None


def _parse_risk_change(
    item: dict[str, Any],
    current: FilingSection,
    previous: FilingSection,
) -> RiskChange:
    return RiskChange(
        company=current.ticker,
        ticker=current.ticker,
        change_type=RiskChangeType(item["change_type"]),
        risk=str(item.get("risk", "")),
        section="Item 1A",
        current_filing=(
            f"{current.filing_type.value} {current.filed_date.isoformat()}"
        ),
        previous_filing=(
            f"{previous.filing_type.value} {previous.filed_date.isoformat()}"
        ),
        language_shift=str(item.get("language_shift", "")),
        severity_estimate=Severity(item.get("severity_estimate", "MEDIUM")),
        related_tickers=item.get("related_tickers", []),
    )


def _changes_to_signals(
    changes: list[RiskChange],
    section: FilingSection,
) -> list[Signal]:
    signals: list[Signal] = []
    for change in changes:
        direction = _DIRECTION_MAP.get(change.change_type, SignalDirection.NEUTRAL)
        strength = _SEVERITY_STRENGTH.get(change.severity_estimate, 0.5)
        signals.append(
            Signal(
                timestamp=datetime.combine(
                    section.filed_date,
                    datetime.min.time(),
                    tzinfo=UTC,
                ),
                ticker=change.ticker,
                signal_type=SignalType.RISK_CHANGE,
                direction=direction,
                strength=strength,
                confidence=0.8,
                context=(
                    f"{change.change_type.value}: {change.risk}"
                    + (f" ({change.language_shift})" if change.language_shift else "")
                ),
                source_filing="",
                related_tickers=change.related_tickers,
                metadata={
                    "change_type": change.change_type.value,
                    "severity": change.severity_estimate.value,
                    "language_shift": change.language_shift,
                    "current_filing": change.current_filing,
                    "previous_filing": change.previous_filing,
                },
            )
        )
    return signals
