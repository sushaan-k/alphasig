"""Webhook notification sender for sigint signals.

Sends JSON payloads to configured webhook URLs when new signals are
extracted.  Supports basic retry and configurable filtering.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from sigint.models import Signal, SignalDirection, SignalType

logger = structlog.get_logger()


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
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _post(self, payload: dict[str, Any]) -> None:
        """POST a JSON payload to the webhook URL with retry."""
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                self._url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    **self._headers,
                },
            )
            resp.raise_for_status()
            logger.debug(
                "webhook_sent",
                url=self._url,
                status=resp.status_code,
            )

    async def send(self, signals: Sequence[Signal]) -> int:
        """Send notifications for qualifying signals.

        Args:
            signals: Signals to evaluate and potentially send.

        Returns:
            Number of notifications sent.
        """
        sent = 0
        for signal in signals:
            if not self._should_send(signal):
                continue
            payload = {
                "source": "sigint",
                "signal": signal.model_dump(mode="json"),
            }
            try:
                await self._post(payload)
                sent += 1
            except httpx.HTTPError as exc:
                logger.error(
                    "webhook_failed",
                    url=self._url,
                    ticker=signal.ticker,
                    error=str(exc),
                )

        logger.info(
            "webhooks_complete",
            url=self._url,
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
            "source": "sigint",
            "batch": True,
            "count": len(qualifying),
            "signals": [s.model_dump(mode="json") for s in qualifying],
        }
        try:
            await self._post(payload)
            logger.info(
                "webhook_batch_sent",
                url=self._url,
                count=len(qualifying),
            )
            return 1
        except httpx.HTTPError as exc:
            logger.error(
                "webhook_batch_failed",
                url=self._url,
                error=str(exc),
            )
            return 0
