"""Tests for output modules -- Parquet, CSV, Webhook, API."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
import respx

from sigint.models import Signal, SignalDirection, SignalType
from sigint.output.parquet import (
    read_signals_parquet,
    write_signals_csv,
    write_signals_parquet,
)
from sigint.output.webhook import WebhookSender


class TestParquetOutput:
    """Tests for Parquet export."""

    def test_write_and_read_parquet(
        self, sample_signals: list[Signal], tmp_path: Path
    ) -> None:
        out = tmp_path / "test_signals.parquet"
        write_signals_parquet(sample_signals, out)
        assert out.exists()
        assert out.stat().st_size > 0

        restored = read_signals_parquet(out)
        assert len(restored) == len(sample_signals)
        assert restored[0].ticker == sample_signals[0].ticker

    def test_parquet_roundtrip_preserves_types(
        self, sample_signals: list[Signal], tmp_path: Path
    ) -> None:
        out = tmp_path / "roundtrip.parquet"
        write_signals_parquet(sample_signals, out)
        restored = read_signals_parquet(out)

        for orig, rest in zip(sample_signals, restored, strict=True):
            assert orig.signal_type == rest.signal_type
            assert orig.direction == rest.direction
            assert abs(orig.strength - rest.strength) < 1e-6

    def test_empty_signals(self, tmp_path: Path) -> None:
        out = tmp_path / "empty.parquet"
        write_signals_parquet([], out)
        restored = read_signals_parquet(out)
        assert restored == []

    def test_parquet_creates_parent_dirs(
        self, sample_signals: list[Signal], tmp_path: Path
    ) -> None:
        out = tmp_path / "nested" / "dir" / "signals.parquet"
        result = write_signals_parquet(sample_signals, out)
        assert result.exists()
        assert result == out

    def test_parquet_roundtrip_preserves_metadata(
        self, sample_signals: list[Signal], tmp_path: Path
    ) -> None:
        out = tmp_path / "meta.parquet"
        write_signals_parquet(sample_signals, out)
        restored = read_signals_parquet(out)
        for orig, rest in zip(sample_signals, restored, strict=True):
            assert orig.metadata == rest.metadata
            assert orig.related_tickers == rest.related_tickers

    def test_csv_creates_parent_dirs(
        self, sample_signals: list[Signal], tmp_path: Path
    ) -> None:
        out = tmp_path / "nested" / "dir" / "signals.csv"
        result = write_signals_csv(sample_signals, out)
        assert result.exists()

    def test_empty_csv(self, tmp_path: Path) -> None:
        out = tmp_path / "empty.csv"
        write_signals_csv([], out)
        content = out.read_text()
        lines = content.strip().splitlines()
        # Header only, no data rows
        assert len(lines) == 1
        assert "ticker" in lines[0]


class TestCSVOutput:
    """Tests for CSV export."""

    def test_write_csv(self, sample_signals: list[Signal], tmp_path: Path) -> None:
        out = tmp_path / "test_signals.csv"
        write_signals_csv(sample_signals, out)
        assert out.exists()

        content = out.read_text()
        lines = content.strip().splitlines()
        # Header + data rows
        assert len(lines) == len(sample_signals) + 1
        assert "ticker" in lines[0]


# -- Webhook Tests -----------------------------------------------------------


@pytest.fixture
def webhook_signals() -> list[Signal]:
    """Signals for webhook testing with varied strength/type/direction."""
    return [
        Signal(
            timestamp=datetime(2024, 11, 1, tzinfo=UTC),
            ticker="AAPL",
            signal_type=SignalType.SUPPLY_CHAIN,
            direction=SignalDirection.NEUTRAL,
            strength=0.9,
            confidence=0.85,
            context="High strength supply chain signal",
            source_filing="",
        ),
        Signal(
            timestamp=datetime(2024, 11, 1, tzinfo=UTC),
            ticker="MSFT",
            signal_type=SignalType.RISK_CHANGE,
            direction=SignalDirection.BEARISH,
            strength=0.3,
            confidence=0.7,
            context="Low strength risk change",
            source_filing="",
        ),
        Signal(
            timestamp=datetime(2024, 11, 1, tzinfo=UTC),
            ticker="GOOG",
            signal_type=SignalType.M_AND_A,
            direction=SignalDirection.BULLISH,
            strength=0.8,
            confidence=0.9,
            context="M&A signal",
            source_filing="",
        ),
    ]


class TestWebhookSender:
    """Tests for the WebhookSender."""

    def test_should_send_no_filters(self, webhook_signals: list[Signal]) -> None:
        sender = WebhookSender("https://hooks.example.com/test")
        assert all(sender._should_send(s) for s in webhook_signals)

    def test_should_send_min_strength_filter(
        self, webhook_signals: list[Signal]
    ) -> None:
        sender = WebhookSender("https://hooks.example.com/test", min_strength=0.5)
        assert sender._should_send(webhook_signals[0])  # 0.9
        assert not sender._should_send(webhook_signals[1])  # 0.3
        assert sender._should_send(webhook_signals[2])  # 0.8

    def test_should_send_signal_type_filter(
        self, webhook_signals: list[Signal]
    ) -> None:
        sender = WebhookSender(
            "https://hooks.example.com/test",
            signal_types=[SignalType.RISK_CHANGE],
        )
        assert not sender._should_send(webhook_signals[0])
        assert sender._should_send(webhook_signals[1])
        assert not sender._should_send(webhook_signals[2])

    def test_should_send_direction_filter(self, webhook_signals: list[Signal]) -> None:
        sender = WebhookSender(
            "https://hooks.example.com/test",
            directions=[SignalDirection.BULLISH],
        )
        assert not sender._should_send(webhook_signals[0])  # neutral
        assert not sender._should_send(webhook_signals[1])  # bearish
        assert sender._should_send(webhook_signals[2])  # bullish

    def test_should_send_combined_filters(self, webhook_signals: list[Signal]) -> None:
        sender = WebhookSender(
            "https://hooks.example.com/test",
            min_strength=0.5,
            directions=[SignalDirection.BULLISH, SignalDirection.NEUTRAL],
        )
        assert sender._should_send(webhook_signals[0])  # neutral, 0.9
        assert not sender._should_send(webhook_signals[1])  # bearish, 0.3
        assert sender._should_send(webhook_signals[2])  # bullish, 0.8

    @respx.mock
    @pytest.mark.asyncio
    async def test_send_posts_signals(self, webhook_signals: list[Signal]) -> None:
        route = respx.post("https://hooks.example.com/test").respond(200)
        sender = WebhookSender("https://hooks.example.com/test")
        sent = await sender.send(webhook_signals)
        assert sent == 3
        assert route.call_count == 3

    @respx.mock
    @pytest.mark.asyncio
    async def test_send_with_min_strength_filters(
        self, webhook_signals: list[Signal]
    ) -> None:
        respx.post("https://hooks.example.com/test").respond(200)
        sender = WebhookSender("https://hooks.example.com/test", min_strength=0.5)
        sent = await sender.send(webhook_signals)
        assert sent == 2  # Only 0.9 and 0.8 qualify

    @respx.mock
    @pytest.mark.asyncio
    async def test_send_handles_http_error(self, webhook_signals: list[Signal]) -> None:
        respx.post("https://hooks.example.com/test").respond(500)
        sender = WebhookSender("https://hooks.example.com/test")
        sent = await sender.send(webhook_signals[:1])
        assert sent == 0

    @respx.mock
    @pytest.mark.asyncio
    async def test_send_batch_posts_all_qualifying(
        self, webhook_signals: list[Signal]
    ) -> None:
        route = respx.post("https://hooks.example.com/test").respond(200)
        sender = WebhookSender("https://hooks.example.com/test")
        result = await sender.send_batch(webhook_signals)
        assert result == 1
        assert route.call_count == 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_send_batch_no_qualifying_returns_zero(self) -> None:
        sender = WebhookSender("https://hooks.example.com/test", min_strength=0.99)
        low = Signal(
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            ticker="X",
            signal_type=SignalType.SUPPLY_CHAIN,
            direction=SignalDirection.NEUTRAL,
            strength=0.1,
            confidence=0.1,
            context="low",
            source_filing="",
        )
        result = await sender.send_batch([low])
        assert result == 0

    @respx.mock
    @pytest.mark.asyncio
    async def test_send_batch_http_error_returns_zero(
        self, webhook_signals: list[Signal]
    ) -> None:
        respx.post("https://hooks.example.com/test").respond(500)
        sender = WebhookSender("https://hooks.example.com/test")
        result = await sender.send_batch(webhook_signals)
        assert result == 0

    def test_custom_headers(self) -> None:
        sender = WebhookSender(
            "https://hooks.example.com/test",
            headers={"Authorization": "Bearer token123"},
        )
        assert sender._headers["Authorization"] == "Bearer token123"


# -- API Tests ---------------------------------------------------------------


class TestAPIServer:
    """Tests for the FastAPI signal server (build_app only)."""

    @pytest.fixture
    def _install_fastapi(self) -> None:
        """Skip if FastAPI is not installed."""
        pytest.importorskip("fastapi")

    @pytest.mark.usefixtures("_install_fastapi")
    def test_build_app_health(self, sample_signals: list[Signal]) -> None:
        from starlette.testclient import TestClient

        from sigint.output.api import _build_app

        app = _build_app(sample_signals)
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    @pytest.mark.usefixtures("_install_fastapi")
    def test_build_app_get_signals(self, sample_signals: list[Signal]) -> None:
        from starlette.testclient import TestClient

        from sigint.output.api import _build_app

        app = _build_app(sample_signals)
        client = TestClient(app)
        resp = client.get("/signals")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == len(sample_signals)

    @pytest.mark.usefixtures("_install_fastapi")
    def test_build_app_filter_by_ticker(self, sample_signals: list[Signal]) -> None:
        from starlette.testclient import TestClient

        from sigint.output.api import _build_app

        app = _build_app(sample_signals)
        client = TestClient(app)
        resp = client.get("/signals?ticker=AAPL")
        assert resp.status_code == 200
        data = resp.json()
        assert all(s["ticker"] == "AAPL" for s in data)

    @pytest.mark.usefixtures("_install_fastapi")
    def test_build_app_filter_by_type(self, sample_signals: list[Signal]) -> None:
        from starlette.testclient import TestClient

        from sigint.output.api import _build_app

        app = _build_app(sample_signals)
        client = TestClient(app)
        resp = client.get("/signals?signal_type=supply_chain")
        assert resp.status_code == 200
        data = resp.json()
        assert all(s["signal_type"] == "supply_chain" for s in data)

    @pytest.mark.usefixtures("_install_fastapi")
    def test_build_app_filter_by_direction(self, sample_signals: list[Signal]) -> None:
        from starlette.testclient import TestClient

        from sigint.output.api import _build_app

        app = _build_app(sample_signals)
        client = TestClient(app)
        resp = client.get("/signals?direction=bearish")
        assert resp.status_code == 200
        data = resp.json()
        assert all(s["direction"] == "bearish" for s in data)

    @pytest.mark.usefixtures("_install_fastapi")
    def test_build_app_filter_min_strength(self, sample_signals: list[Signal]) -> None:
        from starlette.testclient import TestClient

        from sigint.output.api import _build_app

        app = _build_app(sample_signals)
        client = TestClient(app)
        resp = client.get("/signals?min_strength=0.8")
        assert resp.status_code == 200
        data = resp.json()
        assert all(s["strength"] >= 0.8 for s in data)

    @pytest.mark.usefixtures("_install_fastapi")
    def test_build_app_limit(self, sample_signals: list[Signal]) -> None:
        from starlette.testclient import TestClient

        from sigint.output.api import _build_app

        app = _build_app(sample_signals)
        client = TestClient(app)
        resp = client.get("/signals?limit=1")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1

    @pytest.mark.usefixtures("_install_fastapi")
    def test_build_app_summary(self, sample_signals: list[Signal]) -> None:
        from starlette.testclient import TestClient

        from sigint.output.api import _build_app

        app = _build_app(sample_signals)
        client = TestClient(app)
        resp = client.get("/signals/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == len(sample_signals)
        assert "by_type" in data
        assert "by_ticker" in data
        assert "by_direction" in data

    @pytest.mark.usefixtures("_install_fastapi")
    def test_build_app_signals_for_ticker(self, sample_signals: list[Signal]) -> None:
        from starlette.testclient import TestClient

        from sigint.output.api import _build_app

        app = _build_app(sample_signals)
        client = TestClient(app)
        resp = client.get("/signals/AAPL")
        assert resp.status_code == 200
        data = resp.json()
        assert all(s["ticker"] == "AAPL" for s in data)

    @pytest.mark.usefixtures("_install_fastapi")
    def test_build_app_signals_for_ticker_with_type(
        self, sample_signals: list[Signal]
    ) -> None:
        from starlette.testclient import TestClient

        from sigint.output.api import _build_app

        app = _build_app(sample_signals)
        client = TestClient(app)
        resp = client.get("/signals/AAPL?signal_type=supply_chain")
        assert resp.status_code == 200
        data = resp.json()
        assert all(
            s["ticker"] == "AAPL" and s["signal_type"] == "supply_chain" for s in data
        )

    @pytest.mark.usefixtures("_install_fastapi")
    def test_build_app_empty_signals(self) -> None:
        from starlette.testclient import TestClient

        from sigint.output.api import _build_app

        app = _build_app([])
        client = TestClient(app)
        resp = client.get("/signals")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.usefixtures("_install_fastapi")
    def test_build_app_min_confidence_filter(
        self, sample_signals: list[Signal]
    ) -> None:
        from starlette.testclient import TestClient

        from sigint.output.api import _build_app

        app = _build_app(sample_signals)
        client = TestClient(app)
        resp = client.get("/signals?min_confidence=0.9")
        assert resp.status_code == 200
        data = resp.json()
        assert all(s["confidence"] >= 0.9 for s in data)
