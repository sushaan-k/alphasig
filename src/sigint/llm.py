"""LLM client abstraction for structured extraction.

Wraps the Anthropic SDK to provide JSON-schema-constrained extraction
with automatic retry, token accounting, and context-window guards.
"""

from __future__ import annotations

import json
from typing import Any, TypeVar

import anthropic
import structlog
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from sigint.exceptions import LLMContextLengthError, LLMError, LLMRateLimitError

logger = structlog.get_logger()

T = TypeVar("T", bound=BaseModel)

_DEFAULT_MODEL = "claude-sonnet-4-6"
_MAX_OUTPUT_TOKENS = 4096


class LLMClient:
    """Thin wrapper around the Anthropic Messages API.

    Args:
        api_key: Anthropic API key.  Read from ``ANTHROPIC_API_KEY``
            environment variable when *None*.
        model: Model identifier, e.g. ``"claude-sonnet-4-6"``.
        max_output_tokens: Maximum tokens the model may generate.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        max_output_tokens: int = _MAX_OUTPUT_TOKENS,
    ) -> None:
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    @retry(
        retry=retry_if_exception_type(LLMRateLimitError),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    async def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
    ) -> str:
        """Send a single-turn message and return the assistant text.

        Args:
            system: System prompt.
            user: User message content.
            temperature: Sampling temperature.

        Returns:
            The assistant's text response.

        Raises:
            LLMRateLimitError: On 429 responses (retried automatically).
            LLMContextLengthError: If the prompt exceeds the context window.
            LLMError: On any other API error.
        """
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=self._max_output_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except anthropic.RateLimitError as exc:
            raise LLMRateLimitError(str(exc)) from exc
        except anthropic.BadRequestError as exc:
            if "context" in str(exc).lower() or "token" in str(exc).lower():
                raise LLMContextLengthError(str(exc)) from exc
            raise LLMError(str(exc)) from exc
        except anthropic.APIError as exc:
            raise LLMError(str(exc)) from exc

        text_blocks = [b.text for b in response.content if b.type == "text"]
        result = "\n".join(text_blocks)

        logger.debug(
            "llm_response",
            model=self._model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return result

    async def extract_json(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
    ) -> Any:
        """Send a prompt and parse the response as JSON.

        The system prompt should instruct the model to respond with
        valid JSON only.

        Args:
            system: System prompt (should request JSON output).
            user: User message.
            temperature: Sampling temperature.

        Returns:
            Parsed JSON (typically a dict or list).

        Raises:
            LLMError: If the response is not valid JSON.
        """
        raw = await self.complete(system, user, temperature=temperature)

        # Strip markdown fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            # Remove first and last lines (the fences)
            lines = [line for line in lines if not line.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error(
                "llm_json_parse_failed",
                raw_response=raw[:500],
            )
            raise LLMError(f"Model returned invalid JSON: {exc}") from exc

    async def extract_model(
        self,
        response_model: type[T],
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
    ) -> T:
        """Extract structured data into a Pydantic model.

        The system prompt is augmented with the model's JSON schema so
        the LLM knows the expected output format.

        Args:
            response_model: A Pydantic model class.
            system: Base system prompt.
            user: User message.
            temperature: Sampling temperature.

        Returns:
            Validated instance of *response_model*.

        Raises:
            LLMError: On parse or validation failure.
        """
        schema = json.dumps(response_model.model_json_schema(), indent=2)
        augmented_system = (
            f"{system}\n\n"
            f"Respond with valid JSON matching this schema:\n"
            f"```json\n{schema}\n```\n"
            f"Do not include any text outside the JSON object."
        )
        data = await self.extract_json(augmented_system, user, temperature=temperature)
        try:
            return response_model.model_validate(data)
        except Exception as exc:
            raise LLMError(
                f"Failed to validate LLM output against "
                f"{response_model.__name__}: {exc}"
            ) from exc

    async def extract_model_list(
        self,
        response_model: type[T],
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
    ) -> list[T]:
        """Extract a JSON array of Pydantic model instances.

        Args:
            response_model: A Pydantic model class.
            system: Base system prompt.
            user: User message.
            temperature: Sampling temperature.

        Returns:
            List of validated *response_model* instances.
        """
        schema = json.dumps(response_model.model_json_schema(), indent=2)
        augmented_system = (
            f"{system}\n\n"
            f"Respond with a JSON array where each element matches "
            f"this schema:\n```json\n{schema}\n```\n"
            f"Do not include any text outside the JSON array. "
            f"If there are no results, return an empty array []."
        )
        data = await self.extract_json(augmented_system, user, temperature=temperature)
        if not isinstance(data, list):
            data = [data]
        results: list[T] = []
        for item in data:
            try:
                results.append(response_model.model_validate(item))
            except Exception:
                logger.warning(
                    "llm_item_validation_failed",
                    model=response_model.__name__,
                    item=str(item)[:200],
                )
        return results
