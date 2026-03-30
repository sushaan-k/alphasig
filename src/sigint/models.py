"""Pydantic data models used throughout the sigint pipeline.

Every model is immutable (``frozen=True``) so instances are hashable and
safe to share across async tasks.
"""

from __future__ import annotations

import enum
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FilingType(enum.StrEnum):
    """SEC filing types supported by sigint."""

    TEN_K = "10-K"
    TEN_Q = "10-Q"
    EIGHT_K = "8-K"
    DEF_14A = "DEF 14A"


class SignalType(enum.StrEnum):
    """Categories of extracted signals."""

    SUPPLY_CHAIN = "supply_chain"
    RISK_CHANGE = "risk_change"
    M_AND_A = "m_and_a"
    TONE_SHIFT = "tone_shift"


class SignalDirection(enum.StrEnum):
    """Directional bias of a signal."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class RiskChangeType(enum.StrEnum):
    """Classification of how a risk factor changed between filings."""

    NEW = "NEW"
    REMOVED = "REMOVED"
    ESCALATED = "ESCALATED"
    DE_ESCALATED = "DE_ESCALATED"


class Severity(enum.StrEnum):
    """Estimated severity of a risk-factor change."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ToneLabel(enum.StrEnum):
    """Management tone classifications beyond simple polarity."""

    CONFIDENT_EXPANDING = "confident_expanding"
    OPTIMISTIC_CAUTIOUS = "optimistic_cautious"
    NEUTRAL_FACTUAL = "neutral_factual"
    HEDGING_CAUTIOUS = "hedging_cautious"
    DEFENSIVE_JUSTIFYING = "defensive_justifying"
    PESSIMISTIC_WARNING = "pessimistic_warning"


class RelationType(enum.StrEnum):
    """Type of supply-chain relationship between two entities."""

    DEPENDS_ON = "depends_on"
    SUPPLIES_TO = "supplies_to"
    PARTNERS_WITH = "partners_with"


# ---------------------------------------------------------------------------
# Filing models
# ---------------------------------------------------------------------------


class Filing(BaseModel, frozen=True):
    """Metadata and content for a single SEC filing."""

    accession_number: str = Field(
        ..., description="EDGAR accession number, e.g. 0000320193-23-000106"
    )
    cik: str = Field(..., description="Central Index Key of the filer")
    ticker: str = Field(..., description="Trading ticker symbol")
    company_name: str = Field(..., description="Legal entity name")
    filing_type: FilingType
    filed_date: date
    period_of_report: date
    url: str = Field(..., description="EDGAR filing URL")
    raw_html: str = Field(default="", repr=False)


class FilingSection(BaseModel, frozen=True):
    """A parsed section of a filing (e.g. Risk Factors, MD&A)."""

    filing_accession: str
    ticker: str
    section_name: str = Field(
        ..., description="Canonical section name, e.g. 'Risk Factors'"
    )
    section_key: str = Field(..., description="Normalised key, e.g. 'risk_factors'")
    text: str = Field(..., repr=False)
    filing_type: FilingType
    filed_date: date


# ---------------------------------------------------------------------------
# Signal models
# ---------------------------------------------------------------------------


class Signal(BaseModel, frozen=True):
    """Universal signal schema for backtesting compatibility.

    Every extraction engine emits ``Signal`` instances so downstream
    consumers only need a single schema.
    """

    timestamp: datetime = Field(description="Filing date as UTC datetime")
    ticker: str
    signal_type: SignalType
    direction: SignalDirection
    strength: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    context: str = Field(description="Human-readable explanation")
    source_filing: str = Field(description="EDGAR filing URL")
    related_tickers: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("strength", "confidence")
    @classmethod
    def _clamp_unit(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

    @field_validator("related_tickers")
    @classmethod
    def _normalize_tickers(cls, v: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for ticker in v:
            normalised = ticker.strip().upper()
            if normalised and normalised not in seen:
                seen.add(normalised)
                result.append(normalised)
        return result


# ---------------------------------------------------------------------------
# Supply-chain models
# ---------------------------------------------------------------------------


class SupplyChainEdge(BaseModel, frozen=True):
    """A directed edge in the supply-chain knowledge graph."""

    source: str = Field(description="Ticker of the dependent company")
    target: str = Field(description="Ticker or name of the supplier/partner")
    relation: RelationType
    context: str = Field(description="What the relationship concerns")
    confidence: float = Field(ge=0.0, le=1.0)
    filing_type: FilingType
    filed_date: date


# ---------------------------------------------------------------------------
# Risk-factor models
# ---------------------------------------------------------------------------


class RiskChange(BaseModel, frozen=True):
    """A single risk-factor change between consecutive filings."""

    company: str
    ticker: str
    change_type: RiskChangeType
    risk: str = Field(description="Short description of the risk factor")
    section: str = Field(default="Item 1A")
    current_filing: str = Field(description="Label for the current filing")
    previous_filing: str = Field(default="", description="Label for the prior filing")
    language_shift: str = Field(
        default="",
        description="Quoted language change, e.g. 'may face' -> 'currently subject to'",
    )
    severity_estimate: Severity = Severity.MEDIUM
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    related_tickers: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# M&A models
# ---------------------------------------------------------------------------


class MandAIndicator(BaseModel, frozen=True):
    """A single M&A language indicator extracted from a filing."""

    ticker: str
    indicator: str = Field(description="The specific language or pattern detected")
    category: str = Field(
        description=(
            "Category of M&A signal: strategic_alternatives, "
            "advisor_engagement, cash_positioning, board_change, "
            "related_party"
        )
    )
    excerpt: str = Field(description="Verbatim excerpt from the filing", repr=False)
    confidence: float = Field(ge=0.0, le=1.0)
    filing_type: FilingType
    filed_date: date


# ---------------------------------------------------------------------------
# Tone models
# ---------------------------------------------------------------------------


class TonePoint(BaseModel, frozen=True):
    """A single observation in a tone trajectory."""

    filing_label: str = Field(description="e.g. '10-Q Q1 2025'")
    tone: ToneLabel
    confidence: float = Field(ge=0.0, le=1.0)


class ToneTrajectory(BaseModel, frozen=True):
    """Topic-specific tone trajectory across multiple filings."""

    company: str
    ticker: str
    topic: str
    trajectory: list[TonePoint]
    signal: SignalDirection
    signal_strength: float = Field(ge=0.0, le=1.0)
