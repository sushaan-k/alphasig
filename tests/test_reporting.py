"""Tests for sigint.reporting -- offline signal ranking reports."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sigint.models import Signal, SignalDirection, SignalType
from sigint.reporting import rank_signals, summarize_sector_exposure


def _signal(
    ticker: str,
    direction: SignalDirection,
    strength: float,
    confidence: float,
    *,
    timestamp: datetime | None = None,
    context: str = "test signal",
    related_tickers: list[str] | None = None,
    decay_rate: float = 0.0,
) -> Signal:
    return Signal(
        timestamp=timestamp or datetime(2024, 1, 1, tzinfo=UTC),
        ticker=ticker,
        signal_type=SignalType.RISK_CHANGE,
        direction=direction,
        strength=strength,
        confidence=confidence,
        context=context,
        source_filing="https://sec.gov/example",
        related_tickers=related_tickers or [],
        decay_rate=decay_rate,
    )


def test_rank_signals_orders_by_absolute_directional_score() -> None:
    report = rank_signals(
        [
            _signal("AAPL", SignalDirection.BEARISH, 0.9, 0.9),
            _signal("MSFT", SignalDirection.BULLISH, 0.4, 0.8),
            _signal("NVDA", SignalDirection.NEUTRAL, 1.0, 1.0),
        ]
    )

    assert [ticker.ticker for ticker in report.tickers] == ["AAPL", "MSFT", "NVDA"]
    assert report.tickers[0].direction == "bearish"
    assert report.tickers[0].score == -0.81
    assert report.tickers[2].score == 0.0
    assert report.total_signals == 3


def test_rank_signals_averages_multiple_signals_per_ticker() -> None:
    report = rank_signals(
        [
            _signal("AAPL", SignalDirection.BULLISH, 1.0, 0.8),
            _signal("AAPL", SignalDirection.BEARISH, 0.5, 0.8),
        ]
    )

    score = report.tickers[0]
    assert score.ticker == "AAPL"
    assert score.score == 0.2
    assert score.gross_score == 0.6
    assert score.signal_count == 2
    assert score.bullish_count == 1
    assert score.bearish_count == 1


def test_rank_signals_applies_decay_when_as_of_is_supplied() -> None:
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    as_of = timestamp + timedelta(days=10)

    undecayed = rank_signals(
        [
            _signal(
                "AAPL",
                SignalDirection.BULLISH,
                1.0,
                1.0,
                timestamp=timestamp,
                decay_rate=0.0,
            )
        ]
    )
    decayed = rank_signals(
        [
            _signal(
                "AAPL",
                SignalDirection.BULLISH,
                1.0,
                1.0,
                timestamp=timestamp,
                decay_rate=0.1,
            )
        ],
        as_of=as_of,
    )

    assert undecayed.tickers[0].score == 1.0
    assert 0.36 < decayed.tickers[0].score < 0.37


def test_rank_signals_tracks_contexts_and_related_tickers() -> None:
    report = rank_signals(
        [
            _signal(
                "AAPL",
                SignalDirection.BEARISH,
                0.9,
                0.9,
                context="largest signal",
                related_tickers=["TSMC", "NVDA"],
            ),
            _signal(
                "AAPL",
                SignalDirection.BEARISH,
                0.2,
                0.5,
                context="smaller signal",
                related_tickers=["TSMC"],
            ),
        ],
        top_contexts=1,
    )

    score = report.tickers[0]
    assert score.top_contexts == ["largest signal"]
    assert score.related_tickers == {"NVDA": 1, "TSMC": 2}


def test_report_serializes_to_json_and_markdown() -> None:
    report = rank_signals(
        [_signal("AAPL", SignalDirection.BULLISH, 0.7, 0.8)],
        limit=1,
    )

    data = report.to_dict()
    markdown = report.to_markdown()

    assert data["ticker_count"] == 1
    assert data["tickers"][0]["ticker"] == "AAPL"
    assert "| 1 | AAPL | bullish |" in markdown


def test_summarize_sector_exposure_groups_and_orders_by_sector() -> None:
    report = summarize_sector_exposure(
        [
            _signal("AAPL", SignalDirection.BEARISH, 0.9, 0.9),
            _signal("MSFT", SignalDirection.BEARISH, 0.6, 0.8),
            _signal("JPM", SignalDirection.BULLISH, 0.7, 0.8),
        ]
    )

    assert [sector.sector for sector in report.sectors] == [
        "technology",
        "financials",
    ]
    technology = report.sectors[0]
    assert technology.direction == "bearish"
    assert technology.score == -0.645
    assert technology.gross_score == 0.645
    assert technology.signal_count == 2
    assert technology.ticker_count == 2
    assert technology.top_tickers == ["AAPL", "MSFT"]
    assert report.total_signals == 3


def test_summarize_sector_exposure_can_exclude_unknown_tickers() -> None:
    report = summarize_sector_exposure(
        [
            _signal("AAPL", SignalDirection.BULLISH, 0.8, 0.9),
            _signal("ZZZZZ", SignalDirection.BEARISH, 1.0, 1.0),
        ],
        include_unknown=False,
    )

    assert [sector.sector for sector in report.sectors] == ["technology"]
    assert report.total_signals == 2


def test_summarize_sector_exposure_applies_decay() -> None:
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    as_of = timestamp + timedelta(days=10)

    report = summarize_sector_exposure(
        [
            _signal(
                "AAPL",
                SignalDirection.BULLISH,
                1.0,
                1.0,
                timestamp=timestamp,
                decay_rate=0.1,
            )
        ],
        as_of=as_of,
    )

    assert 0.36 < report.sectors[0].score < 0.37
    assert report.sectors[0].avg_strength == report.sectors[0].score


def test_sector_report_serializes_to_json_and_markdown() -> None:
    report = summarize_sector_exposure(
        [_signal("AAPL", SignalDirection.BULLISH, 0.7, 0.8)],
        limit=1,
    )

    data = report.to_dict()
    markdown = report.to_markdown()

    assert data["sector_count"] == 1
    assert data["sectors"][0]["sector"] == "technology"
    assert "| 1 | technology | bullish |" in markdown
