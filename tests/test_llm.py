"""Tests for alphasig.llm -- the Anthropic client wrapper (no network)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from alphasig.exceptions import LLMError
from alphasig.llm import LLMClient, parse_json_response


def _response(text: str, stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=10, output_tokens=5, cache_read_input_tokens=7
        ),
    )


@pytest.fixture
def client_and_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[LLMClient, list[dict[str, Any]], list[SimpleNamespace]]:
    llm = LLMClient(api_key="test-key", model="claude-sonnet-5")
    calls: list[dict[str, Any]] = []
    replies: list[SimpleNamespace] = []

    async def fake_create(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return replies.pop(0)

    monkeypatch.setattr(llm._client.messages, "create", fake_create)
    return llm, calls, replies


class TestParseJsonResponse:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ('[{"a": 1}]', [{"a": 1}]),
            ('```json\n[{"a": 1}]\n```', [{"a": 1}]),
            ('Here you go:\n```\n{"a": 1}\n```\nDone.', {"a": 1}),
            ('Sure! [{"a": [1, 2]}] Hope that helps.', [{"a": [1, 2]}]),
            ("[]", []),
        ],
    )
    def test_recovers_json(self, raw: str, expected: Any) -> None:
        assert parse_json_response(raw) == expected

    def test_rejects_prose(self) -> None:
        with pytest.raises(ValueError):
            parse_json_response("I could not find anything.")


class TestLLMClient:
    @pytest.mark.asyncio
    async def test_request_shape(self, client_and_calls: Any) -> None:
        llm, calls, replies = client_and_calls
        replies.append(_response('[{"x": 1}]'))
        assert await llm.extract_json("SYSTEM", "USER") == [{"x": 1}]

        (params,) = calls
        # Current models reject sampling parameters: none is sent by default.
        assert "temperature" not in params
        # The shared engine instructions are a cacheable prefix.
        assert params["system"] == [
            {"type": "text", "text": "SYSTEM", "cache_control": {"type": "ephemeral"}}
        ]
        assert params["model"] == "claude-sonnet-5"
        assert (llm.input_tokens, llm.output_tokens, llm.cache_read_tokens) == (
            10,
            5,
            7,
        )

    @pytest.mark.asyncio
    async def test_truncated_response_raises(self, client_and_calls: Any) -> None:
        llm, _, replies = client_and_calls
        replies.append(_response('[{"x": 1}, {"x"', stop_reason="max_tokens"))
        with pytest.raises(LLMError, match="truncated"):
            await llm.extract_json("s", "u")

    @pytest.mark.asyncio
    async def test_refusal_raises(self, client_and_calls: Any) -> None:
        llm, _, replies = client_and_calls
        replies.append(_response("", stop_reason="refusal"))
        with pytest.raises(LLMError, match="declined"):
            await llm.complete("s", "u")

    def test_sdk_retries_configured(self) -> None:
        llm = LLMClient(api_key="test-key")
        assert llm._client.max_retries >= 5
