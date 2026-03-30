"""Abstract base class for extraction engines."""

from __future__ import annotations

import abc
from collections.abc import Sequence

from sigint.llm import LLMClient
from sigint.models import FilingSection, Signal


class BaseEngine(abc.ABC):
    """Contract that every extraction engine must satisfy.

    Engines are stateless callables: they receive parsed filing sections
    and an LLM client, and return zero or more :class:`Signal` instances.
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
