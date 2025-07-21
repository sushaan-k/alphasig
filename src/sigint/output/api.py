"""FastAPI REST server for exposing signals over HTTP.

Launch with::

    signals.to_api(port=8080)

Or standalone::

    sigint serve --port 8080
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import structlog

from sigint.models import Signal

logger = structlog.get_logger()


def _build_app(signals: Sequence[Signal]) -> Any:
    """Construct the FastAPI application.

    Deferred import so ``fastapi`` is only required when actually
    serving (it is an optional dependency).
    """
    try:
        from fastapi import FastAPI, Query
    except ImportError as exc:
        raise ImportError(
            "FastAPI is required for the REST API. "
            "Install it with: pip install sigint[api]"
        ) from exc

    app = FastAPI(
        title="sigint",
        description="Causal signal extraction from SEC filings",
        version="0.1.0",
    )

    signal_dicts = [s.model_dump(mode="json") for s in signals]

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/signals")
    async def get_signals(
        ticker: str | None = Query(None),
        signal_type: str | None = Query(None),
        direction: str | None = Query(None),
        min_strength: float | None = Query(None, ge=0.0, le=1.0),
        min_confidence: float | None = Query(None, ge=0.0, le=1.0),
        limit: int = Query(100, ge=1, le=10000),
    ) -> list[dict[str, Any]]:
        results = signal_dicts
        if ticker:
            results = [s for s in results if s["ticker"] == ticker.upper()]
        if signal_type:
            results = [s for s in results if s["signal_type"] == signal_type]
        if direction:
            results = [s for s in results if s["direction"] == direction]
        if min_strength is not None:
            results = [s for s in results if s["strength"] >= min_strength]
        if min_confidence is not None:
            results = [s for s in results if s["confidence"] >= min_confidence]
        return results[:limit]

    @app.get("/signals/summary")
    async def get_summary() -> dict[str, Any]:
        by_type: dict[str, int] = {}
        by_ticker: dict[str, int] = {}
        by_direction: dict[str, int] = {}
        for s in signal_dicts:
            by_type[s["signal_type"]] = by_type.get(s["signal_type"], 0) + 1
            by_ticker[s["ticker"]] = by_ticker.get(s["ticker"], 0) + 1
            by_direction[s["direction"]] = by_direction.get(s["direction"], 0) + 1
        return {
            "total": len(signal_dicts),
            "by_type": by_type,
            "by_ticker": by_ticker,
            "by_direction": by_direction,
        }

    @app.get("/signals/{ticker}")
    async def get_signals_for_ticker(
        ticker: str,
        signal_type: str | None = Query(None),
    ) -> list[dict[str, Any]]:
        results = [s for s in signal_dicts if s["ticker"] == ticker.upper()]
        if signal_type:
            results = [s for s in results if s["signal_type"] == signal_type]
        return results

    return app


def serve_signals(
    signals: Sequence[Signal],
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> None:
    """Start a uvicorn server exposing signals as a REST API.

    # NOTE: For production deployments, add authentication middleware
    # (e.g. API key validation or OAuth) before exposing on 0.0.0.0.

    Args:
        signals: Signals to serve.
        host: Bind address (default ``127.0.0.1``; use ``0.0.0.0``
            only behind an authenticating reverse proxy).
        port: Bind port.
    """
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "uvicorn is required for the REST API. "
            "Install it with: pip install sigint[api]"
        ) from exc

    app = _build_app(signals)
    logger.info("api_starting", host=host, port=port, signals=len(signals))
    uvicorn.run(app, host=host, port=port)
