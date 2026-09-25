"""Abstract base class for extraction engines."""

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Any

from alphasig.llm import LLMClient
from alphasig.models import FilingSection, Signal


def json_objects(raw: Any) -> list[dict[str, Any]]:
    """Normalise an LLM JSON payload to a list of objects.

    Models occasionally return a bare object instead of an array, or mix
    stray strings into the array; anything that is not a JSON object is
    dropped so per-item parsing never trips over it.
    """
    items = raw if isinstance(raw, list) else [raw]
    return [item for item in items if isinstance(item, dict)]


class BaseEngine(abc.ABC):
    """Contract that every extraction engine must satisfy.

    Engines receive parsed filing sections and an LLM client, and return
    zero or more :class:`Signal` instances.  Signal timestamps are the time
    the source filing became public (:attr:`FilingSection.available_at`),
    never the period end, so signals are safe to use point-in-time.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short identifier for the engine (e.g. ``"supply_chain"``)."""

    @abc.abstractmethod
    async def extract(
        self,
        sections: Sequence[FilingSection],
        llm: LLMClient,
        *,
        previous_sections: Sequence[FilingSection] | None = None,
    ) -> list[Signal]:
        """Run extraction on the given filing sections.

        Args:
            sections: Sections from the *current* filing.
            llm: LLM client for structured extraction.
            previous_sections: Sections from the *prior* filing (same
                company, same filing type).  Required by engines that
                perform cross-filing comparison (risk_differ, tone).

        Returns:
            List of signals produced by this engine.
        """
