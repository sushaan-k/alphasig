"""LLM client abstraction for structured extraction.

Wraps the Anthropic SDK to provide JSON extraction with automatic retry,
bounded concurrency, prompt caching, token accounting, and
context-window guards.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, TypeVar

import anthropic
import structlog
from pydantic import BaseModel

from alphasig.exceptions import LLMContextLengthError, LLMError, LLMRateLimitError

logger = structlog.get_logger()

T = TypeVar("T", bound=BaseModel)

#: Model used when none is given; override per call site or with the
#: ``ALPHASIG_MODEL`` environment variable.
DEFAULT_MODEL = os.environ.get("ALPHASIG_MODEL", "claude-sonnet-5")
# Headroom for the model's (adaptive) thinking plus a long JSON array; a
# truncated response is unparseable, so erring high costs nothing extra.
_MAX_OUTPUT_TOKENS = 16_000
_MAX_CONCURRENCY = 8
# The SDK retries 408/409/429/5xx and connection errors with backoff and
# honours ``retry-after``; a pipeline run should ride out a busy minute.
_MAX_RETRIES = 6
_FENCED_JSON = re.compile(r"```(?:json)?\s*\n(.*?)\n\s*```", re.DOTALL)


def parse_json_response(raw: str) -> Any:
    """Parse JSON from a model response, tolerating fences and surrounding prose.

    Raises:
        ValueError: If no JSON value can be recovered.
    """
    cleaned = raw.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    fenced = _FENCED_JSON.search(cleaned)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    decoder = json.JSONDecoder()
    for idx, char in enumerate(cleaned):
        if char in "[{":
            try:
                value, _ = decoder.raw_decode(cleaned, idx)
                return value
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON value found in model response")


class LLMClient:
    """Thin wrapper around the Anthropic Messages API.

    Args:
        api_key: Anthropic API key.  Read from ``ANTHROPIC_API_KEY``
            environment variable when *None*.
        model: Model identifier, e.g. ``"claude-sonnet-5"``.
        max_output_tokens: Maximum tokens the model may generate.
        max_concurrency: Maximum in-flight requests through this client.

    Attributes:
        input_tokens: Running total of input tokens billed by this client.
        output_tokens: Running total of output tokens billed by this client.
        cache_read_tokens: Running total of input tokens served from the
            prompt cache.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_output_tokens: int = _MAX_OUTPUT_TOKENS,
        max_concurrency: int = _MAX_CONCURRENCY,
    ) -> None:
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key, max_retries=_MAX_RETRIES
        )
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens = 0

    async def aclose(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._client.close()

    async def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
    ) -> str:
        """Send a single-turn message and return the assistant text.

        Args:
            system: System prompt.
            user: User message content.
            temperature: Sampling temperature.  Omitted from the request when
                ``None`` -- current models reject sampling parameters.

        Returns:
            The assistant's text response.

        The system prompt is sent as a cached prefix: engines reuse the same
        instructions for every filing, so repeat calls read it from the
        prompt cache (prefixes below the model's minimum are simply not
        cached).

        Raises:
            LLMRateLimitError: On 429 responses that persist after the SDK's
                automatic retries.
            LLMContextLengthError: If the prompt exceeds the context window.
            LLMError: On any other API error, a refusal, or a response
                truncated at ``max_output_tokens``.
        """
        params: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_output_tokens,
            "system": [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": user}],
        }
        if temperature is not None:
            # Not a typed SDK parameter any more; only older models accept it.
            params["extra_body"] = {"temperature": temperature}
        try:
            async with self._semaphore:
                response = await self._client.messages.create(**params)
        except anthropic.RateLimitError as exc:
            raise LLMRateLimitError(str(exc)) from exc
        except anthropic.BadRequestError as exc:
            message = str(exc).lower()
            if "prompt is too long" in message or "context window" in message:
                raise LLMContextLengthError(str(exc)) from exc
            raise LLMError(str(exc)) from exc
        except anthropic.APIError as exc:
            raise LLMError(str(exc)) from exc

        usage = response.usage
        cache_read = getattr(usage, "cache_read_input_tokens", None) or 0
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cache_read_tokens += cache_read
        logger.debug(
            "llm_response",
            model=self._model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=cache_read,
            stop_reason=response.stop_reason,
        )
        if response.stop_reason == "max_tokens":
            raise LLMError(
                f"Response truncated at max_tokens={self._max_output_tokens}"
            )
        if response.stop_reason == "refusal":
            raise LLMError("Model declined the request (stop_reason=refusal)")

        return "\n".join(b.text for b in response.content if b.type == "text")

    async def extract_json(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
    ) -> Any:
        """Send a prompt and parse the response as JSON.

        The system prompt should instruct the model to respond with
        valid JSON only; markdown fences and surrounding prose are tolerated.

        Args:
            system: System prompt (should request JSON output).
            user: User message.
            temperature: Sampling temperature (see :meth:`complete`).

        Returns:
            Parsed JSON (typically a dict or list).

        Raises:
            LLMError: If the response is not valid JSON.
        """
        raw = await self.complete(system, user, temperature=temperature)
        try:
            return parse_json_response(raw)
        except ValueError as exc:
            logger.error("llm_json_parse_failed", raw_response=raw[:500])
            raise LLMError(f"Model returned invalid JSON: {exc}") from exc

    async def extract_model(
        self,
        response_model: type[T],
        system: str,
        user: str,
        *,
        temperature: float | None = None,
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
        temperature: float | None = None,
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
            except Exception as exc:
                logger.error(
                    "llm_item_validation_failed",
                    model=response_model.__name__,
                    item=str(item)[:200],
                    error=str(exc),
                )
        if not results and data:
            logger.error(
                "llm_extraction_fully_failed",
                model=response_model.__name__,
                item_count=len(data),
            )
        return results
