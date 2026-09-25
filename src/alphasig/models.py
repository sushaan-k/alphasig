"""Pydantic data models used throughout the alphasig pipeline.

Every model is immutable (``frozen=True``) so instances are safe to share
across async tasks.
"""

from __future__ import annotations

import enum
import math
from datetime import UTC, date, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, field_validator

_EASTERN = ZoneInfo("America/New_York")
# EDGAR assigns filings accepted after 17:30 ET the next business day's
# filing date, so a filing dated D is always public by D 17:30 ET.
_EDGAR_FILING_CUTOFF = time(17, 30)


def as_utc(value: datetime) -> datetime:
    """Return *value* as an aware UTC datetime (naive input is taken as UTC)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def public_availability(filed_date: date, accepted_at: datetime | None) -> datetime:
    """Return the earliest UTC time a filing is known to be public.

    Uses the EDGAR acceptance timestamp when available.  Otherwise falls
    back to the filing-date cutoff (17:30 ET), which never precedes the real
    release, so backtests keyed on this timestamp cannot trade on a filing
    before it was published.
    """
    if accepted_at is not None:
        return accepted_at.astimezone(UTC)
    return datetime.combine(
        filed_date, _EDGAR_FILING_CUTOFF, tzinfo=_EASTERN
    ).astimezone(UTC)


def decayed_strength(
    strength: float, decay_rate: float, timestamp: datetime, as_of: datetime
) -> float:
    """Exponentially decay *strength* from *timestamp* to *as_of*.

    ``decay_rate`` is per day; the result is clamped to ``[0, 1]`` and never
    exceeds *strength* (no decay is applied before *timestamp*).
    """
    if decay_rate == 0.0:
        return strength
    days_elapsed = (as_utc(as_of) - as_utc(timestamp)).total_seconds() / 86_400.0
    if days_elapsed <= 0:
        return strength
    return max(0.0, min(1.0, strength * math.exp(-decay_rate * days_elapsed)))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FilingType(enum.StrEnum):
    """SEC filing types supported by alphasig."""

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
    accepted_at: datetime | None = Field(
        default=None, description="EDGAR acceptance time (timezone-aware)"
    )
    raw_html: str = Field(default="", repr=False)

    @property
    def available_at(self) -> datetime:
        """UTC time at which the filing became public (see :func:`public_availability`)."""
        return public_availability(self.filed_date, self.accepted_at)


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
    accepted_at: datetime | None = None

    @property
    def available_at(self) -> datetime:
        """UTC time at which the parent filing became public."""
        return public_availability(self.filed_date, self.accepted_at)


# ---------------------------------------------------------------------------
# Signal models
# ---------------------------------------------------------------------------


class Signal(BaseModel, frozen=True):
    """Universal signal schema for backtesting compatibility.

    Every extraction engine emits ``Signal`` instances so downstream
    consumers only need a single schema.
    """

    timestamp: datetime = Field(
        description=(
            "UTC time the source filing became public (EDGAR acceptance time). "
            "Naive datetimes are interpreted as UTC."
        )
    )
    ticker: str
    signal_type: SignalType
    direction: SignalDirection
    strength: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    context: str = Field(description="Human-readable explanation")
    source_filing: str = Field(description="EDGAR filing URL")
    related_tickers: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    decay_rate: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Exponential decay rate (per day). A value of 0 means no decay. "
            "Typical values: 0.005 (half-life ~139 days), "
            "0.01 (half-life ~69 days)."
        ),
    )

    @field_validator("timestamp")
    @classmethod
    def _to_utc(cls, v: datetime) -> datetime:
        return as_utc(v)

    def current_strength(self, *, as_of: datetime) -> float:
        """Compute decayed signal strength at a given point in time.

        Uses exponential decay: ``strength * exp(-decay_rate * days_elapsed)``.
        If *as_of* is before the signal timestamp the original strength is
        returned (signals cannot grow stronger retroactively).

        Args:
            as_of: The datetime at which to evaluate the signal (naive
                datetimes are interpreted as UTC).

        Returns:
            The decayed strength, clamped to ``[0.0, 1.0]``.
        """
        return decayed_strength(self.strength, self.decay_rate, self.timestamp, as_of)

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
    exposure: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Concentration share stated in the filing (e.g. 0.22 for "
            "'22% of net sales'); None when the filing gives no figure"
        ),
    )
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
