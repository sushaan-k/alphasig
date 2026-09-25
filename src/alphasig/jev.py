"""Calibrated signal confidence with Jev (TypeSafe System One).

Engines report the LLM's own ``confidence``, which is a self-assessment
rather than a probability.  :class:`JevCalibrator` re-scores every signal
against the filing text it came from: each signal's claim becomes one
yes/no question, all questions for a filing's evidence go to Jev in a
single call, and Jev's calibrated probability that the claim is supported
replaces ``confidence``.  The LLM's value is kept as
``metadata["llm_confidence"]``.

Requires the optional ``typesafe-sdk`` dependency
(``pip install 'alphasig[jev]'``) and a ``TYPESAFE_API_KEY``.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import structlog

from alphasig.exceptions import ConfigurationError
from alphasig.models import FilingSection, Signal, SignalType

if TYPE_CHECKING:
    from typesafe_sdk import AsyncTypeSafeClient

logger = structlog.get_logger()

#: Sections each signal type is judged against.  ``None`` means every
#: section of the filing.
_EVIDENCE: dict[SignalType, tuple[str, ...] | None] = {
    SignalType.RISK_CHANGE: ("risk_factors",),
    SignalType.TONE_SHIFT: ("md_and_a",),
    SignalType.SUPPLY_CHAIN: ("business", "risk_factors", "md_and_a"),
    SignalType.M_AND_A: None,
}
#: Signal types whose claims compare the filing with its predecessor.
_COMPARATIVE = {SignalType.RISK_CHANGE, SignalType.TONE_SHIFT}

_SUPPORTED = (
    "The filing text in the state explicitly supports the claim; for claims "
    "about a change, comparing the current and previous filings shows it."
)
_UNSUPPORTED = (
    "The filing text does not support the claim, contradicts it, or is silent on it."
)


class JevCalibrator:
    """Replace LLM self-reported confidence with Jev's calibrated probability.

    Args:
        client: A ``typesafe_sdk.AsyncTypeSafeClient``.  Created on first use
            from ``TYPESAFE_API_KEY`` (and ``TYPESAFE_DEFAULT_MODEL``) when
            *None*; a client created here is closed by :meth:`aclose`.
        model: Jev model override (defaults to the SDK's ``jev-latest``).
        min_confidence: Drop signals whose calibrated confidence falls below
            this value.  ``None`` keeps every signal.
        max_section_chars: Per-section cap on the text sent as evidence.
        timeout: Per-request timeout in seconds for a client created here.
    """

    def __init__(
        self,
        client: AsyncTypeSafeClient | None = None,
        *,
        model: str | None = None,
        min_confidence: float | None = None,
        max_section_chars: int = 60_000,
        timeout: float = 60.0,
    ) -> None:
        if min_confidence is not None and not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        self._client = client
        self._owns_client = client is None
        self._model = model
        self._min_confidence = min_confidence
        self._max_section_chars = max_section_chars
        self._timeout = timeout

    def connect(self) -> AsyncTypeSafeClient:
        """Return the Jev client, creating it on first use.

        Raises:
            ConfigurationError: If the SDK is not installed or no API key
                is configured.
        """
        if self._client is None:
            try:
                from typesafe_sdk import AsyncTypeSafeClient, TypeSafeError
            except ImportError as exc:
                raise ConfigurationError(
                    "Jev calibration needs the TypeSafe SDK: "
                    "pip install 'alphasig[jev]'"
                ) from exc
            try:
                self._client = AsyncTypeSafeClient(timeout=self._timeout)
            except TypeSafeError as exc:
                raise ConfigurationError(str(exc)) from exc
        return self._client

    async def aclose(self) -> None:
        """Close the Jev client if this calibrator created it."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> JevCalibrator:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def calibrate(
        self,
        signals: Sequence[Signal],
        sections: Sequence[FilingSection],
        previous_sections: Sequence[FilingSection] | None = None,
    ) -> list[Signal]:
        """Re-score *signals* extracted from one filing.

        Args:
            signals: Signals produced from *sections*.
            sections: Parsed sections of the filing the signals came from.
            previous_sections: The prior filing's sections, used as evidence
                for comparative signals (risk changes, tone shifts).

        Returns:
            The signals with calibrated ``confidence``, in their original
            order, minus any below ``min_confidence``.  A group whose Jev
            request fails keeps its original confidences.
        """
        self.connect()  # fail fast on missing SDK / API key
        groups: dict[SignalType, list[int]] = defaultdict(list)
        for idx, signal in enumerate(signals):
            groups[signal.signal_type].append(idx)

        calibrated = list(signals)
        results = await asyncio.gather(
            *(
                self._score(
                    [signals[i] for i in indices],
                    sections,
                    previous_sections,
                    signal_type,
                )
                for signal_type, indices in groups.items()
            ),
            return_exceptions=True,
        )
        for indices, probabilities in zip(groups.values(), results, strict=True):
            if isinstance(probabilities, BaseException):
                logger.warning(
                    "jev_calibration_failed",
                    signal_type=signals[indices[0]].signal_type.value,
                    error=str(probabilities),
                )
                continue
            for idx, probability in zip(indices, probabilities, strict=True):
                if probability is not None:
                    calibrated[idx] = _with_confidence(signals[idx], probability)

        if self._min_confidence is None:
            return calibrated
        kept = [s for s in calibrated if s.confidence >= self._min_confidence]
        if len(kept) < len(calibrated):
            logger.info(
                "jev_signals_dropped",
                dropped=len(calibrated) - len(kept),
                min_confidence=self._min_confidence,
            )
        return kept

    async def _score(
        self,
        signals: Sequence[Signal],
        sections: Sequence[FilingSection],
        previous_sections: Sequence[FilingSection] | None,
        signal_type: SignalType,
    ) -> list[float | None]:
        """Ask Jev whether each claim is supported; one request per group."""
        from typesafe_sdk import Noul

        state = self._state(sections, previous_sections, signal_type)
        if state is None:
            return [None] * len(signals)
        questions = {
            f"claim_{i}": Noul(
                instructions=_claim(signal),
                criteria={"true": _SUPPORTED, "false": _UNSUPPORTED},
            )
            for i, signal in enumerate(signals)
        }
        response = await self.connect().system_one(
            state=state, questions=questions, model=self._model
        )
        return [
            answer.noul if (answer := response.nouls.get(name)) else None
            for name in questions
        ]

    def _state(
        self,
        sections: Sequence[FilingSection],
        previous_sections: Sequence[FilingSection] | None,
        signal_type: SignalType,
    ) -> dict[str, Any] | None:
        """Build the evidence state, or ``None`` when there is no evidence."""
        keys = _EVIDENCE.get(signal_type)
        current = self._excerpts(sections, keys)
        if not current:
            return None
        first = sections[0]
        state: dict[str, Any] = {
            "company": first.ticker,
            "filing": f"{first.filing_type.value} filed {first.filed_date.isoformat()}",
            "current_filing": current,
        }
        if signal_type in _COMPARATIVE and previous_sections:
            previous = self._excerpts(previous_sections, keys)
            if previous:
                prior = previous_sections[0]
                state["previous_filing_date"] = prior.filed_date.isoformat()
                state["previous_filing"] = previous
        return state

    def _excerpts(
        self, sections: Sequence[FilingSection], keys: tuple[str, ...] | None
    ) -> dict[str, str]:
        return {
            s.section_name: s.text[: self._max_section_chars]
            for s in sections
            if keys is None or s.section_key in keys
        }


def _claim(signal: Signal) -> str:
    """Phrase a signal as a checkable claim, with its quoted evidence."""
    claim = (
        f"Claim about {signal.ticker} ({signal.signal_type.value}, "
        f"{signal.direction.value}): {signal.context}"
    )
    quotes = _quotes(signal.metadata)
    if quotes:
        claim += "\nQuoted evidence: " + "; ".join(quotes)
    return claim + "\nIs this claim supported by the filing text?"


def _quotes(metadata: dict[str, Any]) -> list[str]:
    """Collect the evidence each engine records (tone, supply chain, risk, M&A)."""
    quotes: list[Any] = list(metadata.get("key_phrases") or [])
    quotes += [metadata.get("edge_context"), metadata.get("language_shift")]
    quotes += [
        item.get("excerpt")
        for item in metadata.get("indicators") or []
        if isinstance(item, dict)
    ]
    return [str(q) for q in quotes if q]


def _with_confidence(signal: Signal, probability: float) -> Signal:
    probability = min(1.0, max(0.0, probability))
    return signal.model_copy(
        update={
            "confidence": probability,
            "metadata": {
                **signal.metadata,
                "llm_confidence": signal.confidence,
                "confidence_source": "jev",
            },
        }
    )
