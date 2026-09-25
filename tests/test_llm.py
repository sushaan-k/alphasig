"""Tests for alphasig.llm -- the Anthropic client wrapper (no network)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from alphasig.exceptions import LLMError
from alphasig.llm import LLMCache, LLMClient, parse_json_response


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


class _FakeMessages:
    """Echoes the user message; counts calls and peak concurrency."""

    def __init__(self, delay: float = 0.0, stop_reason: str = "end_turn") -> None:
        self.calls: list[dict[str, Any]] = []
        self.inflight = 0
        self.max_inflight = 0
        self.delay = delay
        self.stop_reason = stop_reason

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.inflight -= 1
        return _response(
            f"echo:{kwargs['messages'][0]['content']}", stop_reason=self.stop_reason
        )


def _fake_client(**kwargs: Any) -> tuple[LLMClient, _FakeMessages]:
    fake = _FakeMessages(
        kwargs.pop("delay", 0.0), kwargs.pop("stop_reason", "end_turn")
    )
    llm = LLMClient(api_key="test-key", **kwargs)
    llm._client = SimpleNamespace(messages=fake)  # type: ignore[assignment]
    return llm, fake


class TestLLMCache:
    async def test_no_cache_by_default(self) -> None:
        llm, fake = _fake_client()
        await llm.complete("sys", "u")
        await llm.complete("sys", "u")
        assert len(fake.calls) == 2
        assert (llm.api_calls, llm.cache_hits) == (2, 0)

    async def test_repeat_request_is_served_from_cache(self) -> None:
        llm, fake = _fake_client(cache=LLMCache())
        assert await llm.complete("sys", "u") == "echo:u"
        assert await llm.complete("sys", "u") == "echo:u"
        assert len(fake.calls) == 1
        assert (llm.api_calls, llm.cache_hits) == (1, 1)
        # Cache hits are not billed.
        assert llm.input_tokens == 10

    async def test_concurrent_identical_requests_share_one_call(self) -> None:
        llm, fake = _fake_client(cache=LLMCache(), delay=0.02)
        results = await asyncio.gather(*[llm.complete("sys", "same") for _ in range(5)])
        assert results == ["echo:same"] * 5
        assert len(fake.calls) == 1

    async def test_key_covers_model_prompts_and_params(self) -> None:
        llm_a, fake_a = _fake_client(cache=LLMCache(), model="model-a")
        await llm_a.complete("sys", "u")
        await llm_a.complete("sys2", "u")  # system prompt differs
        await llm_a.complete("sys", "u2")  # user message differs
        await llm_a.complete("sys", "u", temperature=0.0)  # extra param
        await llm_a.complete("sys", "u", temperature=0.5)
        assert len(fake_a.calls) == 5
        assert len({LLMCache.key(c) for c in fake_a.calls}) == 5

        shared = LLMCache()
        llm_b, fake_b = _fake_client(cache=shared, model="model-b")
        llm_c, fake_c = _fake_client(
            cache=shared, model="model-b", max_output_tokens=100
        )
        await llm_b.complete("sys", "u")
        await llm_c.complete("sys", "u")  # max_tokens differs
        assert (len(fake_b.calls), len(fake_c.calls)) == (1, 1)

    async def test_key_is_stable_across_dict_order(self) -> None:
        assert LLMCache.key({"a": 1, "b": [1, 2]}) == LLMCache.key(
            {"b": [1, 2], "a": 1}
        )

    async def test_disk_cache_persists_across_clients(self, tmp_path: Path) -> None:
        llm1, fake1 = _fake_client(cache=LLMCache(tmp_path))
        await llm1.complete("sys", "u")
        llm2, fake2 = _fake_client(cache=LLMCache(tmp_path))
        assert await llm2.complete("sys", "u") == "echo:u"
        assert (len(fake1.calls), len(fake2.calls)) == (1, 0)
        assert llm2.cache_hits == 1
        assert not list(tmp_path.rglob("*.tmp"))

    async def test_failed_responses_are_not_cached(self, tmp_path: Path) -> None:
        llm, fake = _fake_client(cache=LLMCache(tmp_path), stop_reason="refusal")
        for _ in range(2):
            with pytest.raises(LLMError, match="declined"):
                await llm.complete("sys", "u")
        assert len(fake.calls) == 2
        assert not list(tmp_path.rglob("*.json"))

    async def test_corrupt_disk_entry_is_a_miss(self, tmp_path: Path) -> None:
        cache = LLMCache(tmp_path)
        key = LLMCache.key({"model": "m"})
        path = tmp_path / key[:2] / f"{key}.json"
        path.parent.mkdir(parents=True)
        path.write_text("{truncated")
        assert cache.get(key) is None


class TestConcurrencyLimit:
    @pytest.mark.parametrize("limit", [1, 3])
    async def test_in_flight_requests_are_bounded(self, limit: int) -> None:
        llm, fake = _fake_client(max_concurrency=limit, delay=0.01)
        await asyncio.gather(*[llm.complete("sys", f"u{i}") for i in range(10)])
        assert len(fake.calls) == 10
        assert fake.max_inflight == limit

    def test_rejects_non_positive_limit(self) -> None:
        with pytest.raises(ValueError, match="max_concurrency"):
            LLMClient(api_key="test-key", max_concurrency=0)
