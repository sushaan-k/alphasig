"""Offline reporting helpers for extracted signals."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sigint.models import Signal, SignalDirection
from sigint.sectors import Sector, classify_sector


@dataclass(frozen=True)
class TickerScore:
    """Aggregate score for one ticker in a signal ranking report."""

    ticker: str
    score: float
    gross_score: float
    signal_count: int
    bullish_count: int
    bearish_count: int
    neutral_count: int
    avg_strength: float
    avg_confidence: float
    latest_timestamp: datetime
    top_contexts: list[str]
    related_tickers: dict[str, int]

    @property
    def direction(self) -> str:
        """Human-readable direction implied by the net score."""
        if self.score > 0.0001:
            return "bullish"
        if self.score < -0.0001:
            return "bearish"
        return "neutral"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "ticker": self.ticker,
            "direction": self.direction,
            "score": self.score,
            "gross_score": self.gross_score,
            "signal_count": self.signal_count,
            "bullish_count": self.bullish_count,
            "bearish_count": self.bearish_count,
            "neutral_count": self.neutral_count,
            "avg_strength": self.avg_strength,
            "avg_confidence": self.avg_confidence,
            "latest_timestamp": self.latest_timestamp.isoformat(),
            "top_contexts": self.top_contexts,
            "related_tickers": self.related_tickers,
        }


@dataclass(frozen=True)
class SignalRankingReport:
    """A deterministic, ticker-level ranking of extracted signals."""

    generated_at: datetime
    as_of: datetime | None
    total_signals: int
    tickers: list[TickerScore]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "generated_at": self.generated_at.isoformat(),
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "total_signals": self.total_signals,
            "ticker_count": len(self.tickers),
            "tickers": [ticker.to_dict() for ticker in self.tickers],
        }

    def to_json(self) -> str:
        """Render the report as pretty JSON."""
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self) -> str:
        """Render the report as a compact Markdown table."""
        lines = [
            "# Signal Ranking Report",
            "",
            f"- Generated: {self.generated_at.isoformat()}",
            f"- As of: {self.as_of.isoformat() if self.as_of else 'latest signal'}",
            f"- Signals scored: {self.total_signals}",
            "",
            "| Rank | Ticker | Direction | Score | Gross | Signals | Latest |",
            "|---:|---|---|---:|---:|---:|---|",
        ]
        for idx, ticker in enumerate(self.tickers, start=1):
            lines.append(
                "| "
                f"{idx} | {ticker.ticker} | {ticker.direction} | "
                f"{ticker.score:.4f} | {ticker.gross_score:.4f} | "
                f"{ticker.signal_count} | {ticker.latest_timestamp.date().isoformat()} |"
            )
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class SectorScore:
    """Aggregate directional exposure for one sector."""

    sector: str
    score: float
    gross_score: float
    signal_count: int
    ticker_count: int
    bullish_count: int
    bearish_count: int
    neutral_count: int
    avg_strength: float
    avg_confidence: float
    latest_timestamp: datetime
    top_tickers: list[str]

    @property
    def direction(self) -> str:
        """Human-readable direction implied by the sector net score."""
        if self.score > 0.0001:
            return "bullish"
        if self.score < -0.0001:
            return "bearish"
        return "neutral"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "sector": self.sector,
            "direction": self.direction,
            "score": self.score,
            "gross_score": self.gross_score,
            "signal_count": self.signal_count,
            "ticker_count": self.ticker_count,
            "bullish_count": self.bullish_count,
            "bearish_count": self.bearish_count,
            "neutral_count": self.neutral_count,
            "avg_strength": self.avg_strength,
            "avg_confidence": self.avg_confidence,
            "latest_timestamp": self.latest_timestamp.isoformat(),
            "top_tickers": self.top_tickers,
        }


@dataclass(frozen=True)
class SectorExposureReport:
    """A deterministic sector-level exposure summary of extracted signals."""

    generated_at: datetime
    as_of: datetime | None
    total_signals: int
    sectors: list[SectorScore]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "generated_at": self.generated_at.isoformat(),
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "total_signals": self.total_signals,
            "sector_count": len(self.sectors),
            "sectors": [sector.to_dict() for sector in self.sectors],
        }

    def to_json(self) -> str:
        """Render the report as pretty JSON."""
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self) -> str:
        """Render the report as a compact Markdown table."""
        lines = [
            "# Sector Exposure Report",
            "",
            f"- Generated: {self.generated_at.isoformat()}",
            f"- As of: {self.as_of.isoformat() if self.as_of else 'latest signal'}",
            f"- Signals scored: {self.total_signals}",
            "",
            "| Rank | Sector | Direction | Score | Gross | Tickers | Signals | Latest |",
            "|---:|---|---|---:|---:|---:|---:|---|",
        ]
        for idx, sector in enumerate(self.sectors, start=1):
            lines.append(
                "| "
                f"{idx} | {sector.sector} | {sector.direction} | "
                f"{sector.score:.4f} | {sector.gross_score:.4f} | "
                f"{sector.ticker_count} | {sector.signal_count} | "
                f"{sector.latest_timestamp.date().isoformat()} |"
            )
        return "\n".join(lines) + "\n"


def rank_signals(
    signals: Iterable[Signal],
    *,
    as_of: datetime | None = None,
    limit: int | None = None,
    top_contexts: int = 3,
) -> SignalRankingReport:
    """Rank tickers by confidence-weighted directional signal strength.

    Bullish signals contribute positively, bearish signals negatively, and
    neutral signals add to gross exposure but not net direction.  Scores are
    averaged per ticker so a ticker with many weak duplicate signals does not
    automatically outrank a ticker with fewer, stronger signals.
    """
    signal_list = list(signals)
    grouped: dict[str, list[Signal]] = {}
    for signal in signal_list:
        grouped.setdefault(signal.ticker.upper(), []).append(signal)

    tickers = [
        _score_ticker(
            ticker=ticker,
            signals=ticker_signals,
            as_of=as_of,
            top_contexts=top_contexts,
        )
        for ticker, ticker_signals in grouped.items()
    ]
    tickers.sort(
        key=lambda item: (
            abs(item.score),
            item.gross_score,
            item.latest_timestamp,
            item.ticker,
        ),
        reverse=True,
    )
    if limit is not None:
        tickers = tickers[:limit]

    return SignalRankingReport(
        generated_at=datetime.now(UTC),
        as_of=as_of,
        total_signals=len(signal_list),
        tickers=tickers,
    )


def summarize_sector_exposure(
    signals: Iterable[Signal],
    *,
    as_of: datetime | None = None,
    limit: int | None = None,
    include_unknown: bool = True,
    top_tickers: int = 3,
) -> SectorExposureReport:
    """Summarize confidence-weighted directional exposure by sector.

    This mirrors :func:`rank_signals` at the portfolio level: bullish signals
    add positive exposure, bearish signals add negative exposure, and neutral
    signals contribute to gross exposure. Scores are averaged per signal inside
    each sector, keeping broad sectors with many duplicate signals from
    dominating the report solely by count.
    """
    signal_list = list(signals)
    grouped: dict[Sector, list[Signal]] = {}
    for signal in signal_list:
        sector = classify_sector(signal.ticker)
        if sector is Sector.UNKNOWN and not include_unknown:
            continue
        grouped.setdefault(sector, []).append(signal)

    sectors = [
        _score_sector(
            sector=sector,
            signals=sector_signals,
            as_of=as_of,
            top_tickers=top_tickers,
        )
        for sector, sector_signals in grouped.items()
    ]
    sectors.sort(
        key=lambda item: (
            abs(item.score),
            item.gross_score,
            item.signal_count,
            item.sector,
        ),
        reverse=True,
    )
    if limit is not None:
        sectors = sectors[:limit]

    return SectorExposureReport(
        generated_at=datetime.now(UTC),
        as_of=as_of,
        total_signals=len(signal_list),
        sectors=sectors,
    )


def _score_ticker(
    *,
    ticker: str,
    signals: list[Signal],
    as_of: datetime | None,
    top_contexts: int,
) -> TickerScore:
    components: list[float] = []
    gross_components: list[float] = []
    strengths: list[float] = []
    confidences: list[float] = []
    contexts: list[tuple[float, str]] = []
    related: Counter[str] = Counter()

    bullish = bearish = neutral = 0
    latest = max(signal.timestamp for signal in signals)

    for signal in signals:
        direction_weight = _direction_weight(signal.direction)
        if signal.direction is SignalDirection.BULLISH:
            bullish += 1
        elif signal.direction is SignalDirection.BEARISH:
            bearish += 1
        else:
            neutral += 1

        effective_strength = (
            signal.current_strength(as_of=as_of) if as_of else signal.strength
        )
        weighted = effective_strength * signal.confidence
        components.append(direction_weight * weighted)
        gross_components.append(weighted)
        strengths.append(effective_strength)
        confidences.append(signal.confidence)
        contexts.append((weighted, signal.context))
        related.update(signal.related_tickers)

    count = len(signals)
    contexts.sort(key=lambda item: item[0], reverse=True)

    return TickerScore(
        ticker=ticker,
        score=round(sum(components) / count, 6),
        gross_score=round(sum(gross_components) / count, 6),
        signal_count=count,
        bullish_count=bullish,
        bearish_count=bearish,
        neutral_count=neutral,
        avg_strength=round(sum(strengths) / count, 6),
        avg_confidence=round(sum(confidences) / count, 6),
        latest_timestamp=latest,
        top_contexts=[context for _, context in contexts[:top_contexts]],
        related_tickers=dict(sorted(related.items())),
    )


def _score_sector(
    *,
    sector: Sector,
    signals: list[Signal],
    as_of: datetime | None,
    top_tickers: int,
) -> SectorScore:
    components: list[float] = []
    gross_components: list[float] = []
    strengths: list[float] = []
    confidences: list[float] = []
    ticker_components: dict[str, list[float]] = {}

    bullish = bearish = neutral = 0
    latest = max(signal.timestamp for signal in signals)

    for signal in signals:
        direction_weight = _direction_weight(signal.direction)
        if signal.direction is SignalDirection.BULLISH:
            bullish += 1
        elif signal.direction is SignalDirection.BEARISH:
            bearish += 1
        else:
            neutral += 1

        effective_strength = (
            signal.current_strength(as_of=as_of) if as_of else signal.strength
        )
        weighted = effective_strength * signal.confidence
        signed_weighted = direction_weight * weighted
        components.append(signed_weighted)
        gross_components.append(weighted)
        strengths.append(effective_strength)
        confidences.append(signal.confidence)
        ticker_components.setdefault(signal.ticker.upper(), []).append(signed_weighted)

    count = len(signals)
    ranked_tickers = sorted(
        ticker_components.items(),
        key=lambda item: (
            abs(sum(item[1]) / len(item[1])),
            sum(abs(component) for component in item[1]) / len(item[1]),
            item[0],
        ),
        reverse=True,
    )

    return SectorScore(
        sector=sector.value,
        score=round(sum(components) / count, 6),
        gross_score=round(sum(gross_components) / count, 6),
        signal_count=count,
        ticker_count=len(ticker_components),
        bullish_count=bullish,
        bearish_count=bearish,
        neutral_count=neutral,
        avg_strength=round(sum(strengths) / count, 6),
        avg_confidence=round(sum(confidences) / count, 6),
        latest_timestamp=latest,
        top_tickers=[ticker for ticker, _ in ranked_tickers[:top_tickers]],
    )


def _direction_weight(direction: SignalDirection) -> float:
    if direction is SignalDirection.BULLISH:
        return 1.0
    if direction is SignalDirection.BEARISH:
        return -1.0
    return 0.0
