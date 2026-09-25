"""Tests for Jev calibrated confidence (TypeSafe System One)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx2
import pytest
import respx
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from alphasig.exceptions import ConfigurationError
from alphasig.jev import JevCalibrator
from alphasig.models import FilingSection, Signal, SignalDirection, SignalType
from alphasig.pipeline import Pipeline


def _signal(signal_type: SignalType, context: str, **metadata: Any) -> Signal:
    return Signal(
        timestamp=datetime(2024, 11, 1, 22, 4, tzinfo=UTC),
        ticker="AAPL",
        signal_type=signal_type,
        direction=SignalDirection.BEARISH,
        strength=0.6,
        confidence=0.9,
        context=context,
        source_filing="",
        metadata=metadata,
    )


class _FakeJev:
    """A Jev endpoint answering every question with a fixed probability."""

    def __init__(self, probability: float = 0.25, status: int = 200) -> None:
        self.probability = probability
        self.status = status
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        if self.status != 200:
            return httpx2.Response(self.status, json={"detail": "boom"})
        answers = {
            name: {"type": "noul", "noul": self.probability}
            for name in body["questions"]
        }
        return httpx2.Response(
            200,
            json={
                "model": "jev-latest",
                "usage": {"input_tokens": 100, "output_tokens": 1},
                "answers": answers,
            },
        )

    def client(self) -> AsyncTypeSafeClient:
        return AsyncTypeSafeClient(
            api_key="test-key",
            transport=httpx2.MockTransport(self),
            retry=RetryPolicy(max_retries=0),
        )


class TestJevCalibrator:
    @pytest.mark.asyncio
    async def test_replaces_confidence_and_keeps_llm_value(
        self,
        sample_sections: list[FilingSection],
        previous_sections: list[FilingSection],
    ) -> None:
        jev = _FakeJev(probability=0.25)
        signals = [
            _signal(
                SignalType.RISK_CHANGE,
                "New risk: EU regulatory scrutiny",
                language_shift="may face -> face",
            ),
            _signal(
                SignalType.SUPPLY_CHAIN,
                "AAPL depends on TSM",
                edge_context="relies on TSMC",
            ),
            _signal(SignalType.RISK_CHANGE, "Escalated: supply concentration"),
        ]
        async with JevCalibrator(jev.client()) as calibrator:
            result = await calibrator.calibrate(
                signals, sample_sections, previous_sections
            )

        assert [s.context for s in result] == [s.context for s in signals]
        assert all(s.confidence == 0.25 for s in result)
        assert all(s.metadata["llm_confidence"] == 0.9 for s in result)
        assert all(s.metadata["confidence_source"] == "jev" for s in result)

        # One request per signal type, one question per signal.
        by_questions = {len(r["questions"]): r for r in jev.requests}
        assert sorted(by_questions) == [1, 2]
        risk = by_questions[2]
        assert set(risk["state"]["current_filing"]) == {"Risk Factors"}
        assert set(risk["state"]["previous_filing"]) == {"Risk Factors"}
        assert risk["questions"]["claim_0"]["type"] == "noul"
        assert "may face -> face" in risk["questions"]["claim_0"]["instructions"]
        supply = by_questions[1]
        assert set(supply["state"]["current_filing"]) == {
            "Business",
            "Risk Factors",
            "Management Discussion and Analysis",
        }
        assert "previous_filing" not in supply["state"]

    @pytest.mark.asyncio
    async def test_min_confidence_drops_unsupported_claims(
        self, sample_sections: list[FilingSection]
    ) -> None:
        jev = _FakeJev(probability=0.1)
        calibrator = JevCalibrator(jev.client(), min_confidence=0.5)
        signals = [_signal(SignalType.M_AND_A, "Acquisition language")]
        assert await calibrator.calibrate(signals, sample_sections) == []

    @pytest.mark.asyncio
    async def test_failed_request_keeps_original_signals(
        self, sample_sections: list[FilingSection]
    ) -> None:
        jev = _FakeJev(status=500)
        signals = [_signal(SignalType.M_AND_A, "Acquisition language")]
        result = await JevCalibrator(jev.client()).calibrate(signals, sample_sections)
        assert result == signals

    @pytest.mark.asyncio
    async def test_no_evidence_leaves_signal_unchanged(
        self, sample_sections: list[FilingSection]
    ) -> None:
        jev = _FakeJev()
        business_only = [s for s in sample_sections if s.section_key == "business"]
        signals = [_signal(SignalType.TONE_SHIFT, "Tone shifted")]
        result = await JevCalibrator(jev.client()).calibrate(signals, business_only)
        assert result == signals
        assert jev.requests == []

    def test_missing_api_key_is_a_configuration_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        with pytest.raises(ConfigurationError, match="TYPESAFE_API_KEY"):
            JevCalibrator().connect()

    def test_rejects_out_of_range_threshold(self) -> None:
        with pytest.raises(ValueError):
            JevCalibrator(min_confidence=1.5)


class TestPipelineCalibration:
    @respx.mock
    @pytest.mark.asyncio
    async def test_pipeline_calibrates_extracted_signals(self) -> None:
        respx.get("https://www.sec.gov/files/company_tickers.json").respond(
            json={"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple"}}
        )
        respx.get("https://data.sec.gov/submissions/CIK0000320193.json").respond(
            json={
                "name": "Apple Inc.",
                "filings": {
                    "recent": {
                        "accessionNumber": ["0000320193-24-000123"],
                        "form": ["10-K"],
                        "filingDate": ["2024-11-01"],
                        "reportDate": ["2024-09-28"],
                        "primaryDocument": ["aapl-20240928.htm"],
                        "acceptanceDateTime": ["2024-11-01T18:04:43.000Z"],
                    }
                },
            }
        )
        filler = "Apple relies on TSMC for chips and on Foxconn for assembly. " * 12
        respx.get(url__startswith="https://www.sec.gov/Archives/").respond(
            text=f"<html><body><p><b>Item 1. Business</b></p><p>{filler}</p></body></html>"
        )
        reply = AsyncMock(
            return_value=[
                {"target": "TSM", "relation": "depends_on", "confidence": 0.9}
            ]
        )
        jev = _FakeJev(probability=0.8)
        with (
            patch("alphasig.llm.LLMClient.extract_json", new=reply),
            patch("alphasig.llm.LLMClient.aclose", new=AsyncMock()),
        ):
            collection = await Pipeline(
                api_key="k",
                user_agent="Test test@example.com",
                cache_dir=None,
                db_path=None,
                calibrator=JevCalibrator(jev.client()),
            ).extract(
                tickers=["AAPL"],
                filing_types=["10-K"],
                engines=["supply_chain"],
                store=False,
            )

        (sig,) = list(collection)
        assert sig.confidence == 0.8
        assert sig.metadata["llm_confidence"] == 0.9
        assert sig.metadata["_filing_accession"] == "0000320193-24-000123"
        assert len(jev.requests) == 1
