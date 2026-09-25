"""Webhook notification sender for alphasig signals.

Sends JSON payloads to configured webhook URLs when new signals are
extracted.  Supports basic retry and configurable filtering.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from alphasig.models import Signal, SignalDirection, SignalType

logger = structlog.get_logger()

_TIMEOUT = httpx.Timeout(15.0)


def _describe(exc: httpx.HTTPError) -> str:
    """Summarise an httpx error without its URL (which may hold a secret)."""
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    return type(exc).__name__


def _is_transient(exc: BaseException) -> bool:
    """Retry network errors, 429 and 5xx; a 4xx will not succeed on retry."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return isinstance(exc, httpx.TransportError)


class WebhookSender:
    """Sends signal notifications to webhook endpoints.

    Args:
        url: The webhook URL to POST to.
        headers: Optional extra headers (e.g. auth tokens).
        min_strength: Only send signals above this strength threshold.
        signal_types: If set, only send these signal types.
        directions: If set, only send signals with these directions.
    """

    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        min_strength: float = 0.0,
        signal_types: Sequence[SignalType] | None = None,
        directions: Sequence[SignalDirection] | None = None,
    ) -> None:
        self._url = url
        # Webhook URLs often embed their secret (Slack, Discord, ...), so
        # logs only ever show the host.
        self._log_target = urlsplit(url).netloc or "webhook"
        self._headers = headers or {}
        self._min_strength = min_strength
        self._signal_types = set(signal_types) if signal_types else None
        self._directions = set(directions) if directions else None

    def _should_send(self, signal: Signal) -> bool:
        """Check whether a signal passes the configured filters."""
        if signal.strength < self._min_strength:
            return False
        if self._signal_types and signal.signal_type not in self._signal_types:
            return False
        return not (self._directions and signal.direction not in self._directions)

    @retry(
        retry=retry_if_exception(_is_transient),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _post(self, client: httpx.AsyncClient, payload: dict[str, Any]) -> None:
        """POST a JSON payload to the webhook URL, retrying transient failures."""
        resp = await client.post(self._url, json=payload, headers=self._headers)
        resp.raise_for_status()
        logger.debug("webhook_sent", target=self._log_target, status=resp.status_code)

    async def send(self, signals: Sequence[Signal]) -> int:
        """Send one notification per qualifying signal.

        Args:
            signals: Signals to evaluate and potentially send.

        Returns:
            Number of notifications sent.
        """
        sent = 0
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            for signal in signals:
                if not self._should_send(signal):
                    continue
                payload = {
                    "source": "alphasig",
                    "signal": signal.model_dump(mode="json"),
                }
                try:
                    await self._post(client, payload)
                    sent += 1
                except httpx.HTTPError as exc:
                    logger.error(
                        "webhook_failed",
                        target=self._log_target,
                        ticker=signal.ticker,
                        error=_describe(exc),
                    )

        logger.info(
            "webhooks_complete",
            target=self._log_target,
            sent=sent,
            total=len(signals),
        )
        return sent

    async def send_batch(self, signals: Sequence[Signal]) -> int:
        """Send all qualifying signals in a single batch payload.

        Args:
            signals: Signals to evaluate.

        Returns:
            1 if the batch was sent, 0 otherwise.
        """
        qualifying = [s for s in signals if self._should_send(s)]
        if not qualifying:
            return 0

        payload = {
            "source": "alphasig",
            "batch": True,
            "count": len(qualifying),
            "signals": [s.model_dump(mode="json") for s in qualifying],
        }
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                await self._post(client, payload)
        except httpx.HTTPError as exc:
            logger.error(
                "webhook_batch_failed",
                target=self._log_target,
                error=_describe(exc),
            )
            return 0
        logger.info(
            "webhook_batch_sent", target=self._log_target, count=len(qualifying)
        )
        return 1
