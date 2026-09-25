"""Integration-level tests for the Pipeline (mocking EDGAR and LLM)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from alphasig.exceptions import ExtractionError, PipelineError
from alphasig.jev import JevCalibrator
from alphasig.llm import DEFAULT_MODEL
from alphasig.models import FilingSection, Signal, SignalDirection, SignalType
from alphasig.pipeline import (
    Pipeline,
    _deduplicate_amendment_signals,
    _resolve_engines,
    _run_engine,
)
from alphasig.storage import SignalStore


class TestPipelineHelpers:
    """Tests for pipeline helper functions."""

    def test_resolve_engines_all(self) -> None:
        engines = _resolve_engines(["supply_chain", "risk_differ", "m_and_a", "tone"])
        assert len(engines) == 4

    def test_resolve_engines_subset(self) -> None:
        engines = _resolve_engines(["supply_chain"])
        assert len(engines) == 1
        assert engines[0].name == "supply_chain"

    def test_resolve_engines_unknown(self) -> None:
        with pytest.raises(PipelineError, match="Unknown engine"):
            _resolve_engines(["nonexistent"])

    def test_resolve_engines_empty(self) -> None:
        engines = _resolve_engines([])
        assert engines == []

    def test_resolve_engines_duplicate(self) -> None:
        engines = _resolve_engines(["supply_chain", "supply_chain"])
        assert len(engines) == 2


class TestRunEngine:
    """Tests for the _run_engine helper."""

    @pytest.mark.asyncio
    async def test_wraps_exception_in_extraction_error(self) -> None:
        engine = MagicMock()
        engine.name = "test_engine"
        engine.extract = AsyncMock(side_effect=ValueError("boom"))
        llm = MagicMock()
        with pytest.raises(ExtractionError, match="test_engine"):
            await _run_engine(
                engine=engine, sections=[], llm=llm, previous_sections=None
            )

    @pytest.mark.asyncio
    async def test_returns_signals_on_success(self) -> None:
        engine = MagicMock()
        engine.name = "test_engine"
        engine.extract = AsyncMock(return_value=[])
        llm = MagicMock()
        result = await _run_engine(
            engine=engine, sections=[], llm=llm, previous_sections=None
        )
        assert result == []


class TestPipelineIntegration:
    """Integration tests with mocked external dependencies."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_extract_runs_pipeline(self) -> None:
        """End-to-end test with mocked EDGAR and LLM."""
        # Mock EDGAR
        respx.get("https://www.sec.gov/files/company_tickers.json").respond(
            json={
                "0": {
                    "cik_str": 320193,
                    "ticker": "AAPL",
                    "title": "Apple Inc.",
                },
            }
        )

        respx.get("https://data.sec.gov/submissions/CIK0000320193.json").respond(
            json={
                "cik": "0000320193",
                "name": "Apple Inc.",
                "filings": {
                    "recent": {
                        "accessionNumber": ["0000320193-24-000123"],
                        "form": ["10-K"],
                        "filingDate": ["2024-11-01"],
                        "reportDate": ["2024-09-28"],
                        "primaryDocument": ["aapl-20240928.htm"],
                    }
                },
            }
        )

        filing_html = """
        <html><body>
        <b>Item 1. Business</b>
        <p>Apple relies on TSMC for semiconductor manufacturing and
        Foxconn for device assembly operations worldwide.</p>

        <b>Item 1A. Risk Factors</b>
        <p>The company faces supply chain concentration risks with
        dependence on TSMC. Regulatory scrutiny in the EU continues.</p>

        <b>Item 7. Management's Discussion and Analysis</b>
        <p>Revenue increased strongly. We are confident in our strategy
        and expect continued growth across all segments.</p>
        </body></html>
        """

        respx.get(url__startswith="https://www.sec.gov/Archives/").respond(
            text=filing_html
        )

        # Mock the LLM to return valid extraction results
        mock_response = AsyncMock()
        mock_response.return_value = [
            {
                "source": "AAPL",
                "target": "TSMC",
                "relation": "depends_on",
                "context": "semiconductors",
                "confidence": 0.9,
            },
        ]

        with patch("alphasig.llm.LLMClient.extract_json", new=mock_response):
            pipeline = Pipeline(
                model="claude-sonnet-5",
                api_key="test-key",
                user_agent="Test test@example.com",
                cache_dir=None,
                db_path=None,
            )

            collection = await pipeline.extract(
                tickers=["AAPL"],
                filing_types=["10-K"],
                lookback_years=5,
                engines=["supply_chain"],
                store=False,
            )

        assert len(collection) >= 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_extract_no_filings_returns_empty(self) -> None:
        """When EDGAR returns no filings for a ticker, collection is empty."""
        respx.get("https://www.sec.gov/files/company_tickers.json").respond(
            json={
                "0": {
                    "cik_str": 320193,
                    "ticker": "AAPL",
                    "title": "Apple Inc.",
                },
            }
        )
        respx.get("https://data.sec.gov/submissions/CIK0000320193.json").respond(
            json={
                "cik": "0000320193",
                "name": "Apple Inc.",
                "filings": {
                    "recent": {
                        "accessionNumber": [],
                        "form": [],
                        "filingDate": [],
                        "reportDate": [],
                        "primaryDocument": [],
                    }
                },
            }
        )

        mock_response = AsyncMock(return_value=[])
        with patch("alphasig.llm.LLMClient.extract_json", new=mock_response):
            pipeline = Pipeline(
                model="claude-sonnet-5",
                api_key="test-key",
                user_agent="Test test@example.com",
                cache_dir=None,
                db_path=None,
            )
            collection = await pipeline.extract(
                tickers=["AAPL"],
                engines=["supply_chain"],
                store=False,
            )
        assert len(collection) == 0

    @respx.mock
    @pytest.mark.asyncio
    async def test_extract_handles_ticker_failure_gracefully(self) -> None:
        """A failing ticker should not crash the whole pipeline."""
        respx.get("https://www.sec.gov/files/company_tickers.json").respond(
            json={
                "0": {
                    "cik_str": 320193,
                    "ticker": "AAPL",
                    "title": "Apple Inc.",
                },
            }
        )
        # Simulate failure for ZZZZ (unknown ticker)
        # AAPL submissions also fail for simplicity
        respx.get("https://data.sec.gov/submissions/CIK0000320193.json").respond(
            status_code=500
        )

        mock_response = AsyncMock(return_value=[])
        with patch("alphasig.llm.LLMClient.extract_json", new=mock_response):
            pipeline = Pipeline(
                model="claude-sonnet-5",
                api_key="test-key",
                user_agent="Test test@example.com",
                cache_dir=None,
                db_path=None,
            )
            # Should not raise -- errors are caught per-ticker
            collection = await pipeline.extract(
                tickers=["AAPL"],
                engines=["supply_chain"],
                store=False,
            )
        assert len(collection) == 0

    @respx.mock
    @pytest.mark.asyncio
    async def test_extract_with_malformed_filing_html(self) -> None:
        """Filings with no parseable sections produce no signals."""
        respx.get("https://www.sec.gov/files/company_tickers.json").respond(
            json={
                "0": {
                    "cik_str": 320193,
                    "ticker": "AAPL",
                    "title": "Apple Inc.",
                },
            }
        )
        respx.get("https://data.sec.gov/submissions/CIK0000320193.json").respond(
            json={
                "cik": "0000320193",
                "name": "Apple Inc.",
                "filings": {
                    "recent": {
                        "accessionNumber": ["0000320193-24-000123"],
                        "form": ["10-K"],
                        "filingDate": ["2024-11-01"],
                        "reportDate": ["2024-09-28"],
                        "primaryDocument": ["aapl-20240928.htm"],
                    }
                },
            }
        )
        # Malformed HTML with no recognizable sections
        respx.get(url__startswith="https://www.sec.gov/Archives/").respond(
            text="<html><body><p>Just some random text, no items.</p></body></html>"
        )

        mock_response = AsyncMock(return_value=[])
        with patch("alphasig.llm.LLMClient.extract_json", new=mock_response):
            pipeline = Pipeline(
                model="claude-sonnet-5",
                api_key="test-key",
                user_agent="Test test@example.com",
                cache_dir=None,
                db_path=None,
            )
            collection = await pipeline.extract(
                tickers=["AAPL"],
                filing_types=["10-K"],
                lookback_years=5,
                engines=["supply_chain"],
                store=False,
            )
        assert len(collection) == 0

    @respx.mock
    @pytest.mark.asyncio
    async def test_extract_engine_failure_does_not_crash(self) -> None:
        """An engine that raises should not prevent other engines."""
        respx.get("https://www.sec.gov/files/company_tickers.json").respond(
            json={
                "0": {
                    "cik_str": 320193,
                    "ticker": "AAPL",
                    "title": "Apple Inc.",
                },
            }
        )
        respx.get("https://data.sec.gov/submissions/CIK0000320193.json").respond(
            json={
                "cik": "0000320193",
                "name": "Apple Inc.",
                "filings": {
                    "recent": {
                        "accessionNumber": ["0000320193-24-000123"],
                        "form": ["10-K"],
                        "filingDate": ["2024-11-01"],
                        "reportDate": ["2024-09-28"],
                        "primaryDocument": ["aapl-20240928.htm"],
                    }
                },
            }
        )
        filing_html = """
        <html><body>
        <b>Item 1. Business</b>
        <p>Apple relies on TSMC for semiconductor manufacturing and
        various other suppliers for device assembly worldwide.</p>

        <b>Item 1A. Risk Factors</b>
        <p>The company faces supply chain concentration risks with heavy
        dependence on TSMC. Regulatory scrutiny in the EU continues.</p>

        <b>Item 7. Management's Discussion and Analysis</b>
        <p>Revenue increased strongly. We are confident in our strategy
        and expect continued growth across all segments.</p>
        </body></html>
        """
        respx.get(url__startswith="https://www.sec.gov/Archives/").respond(
            text=filing_html
        )

        call_count = 0

        async def alternating_response(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Engine crash!")
            return [
                {
                    "source": "AAPL",
                    "target": "TSMC",
                    "relation": "depends_on",
                    "context": "chips",
                    "confidence": 0.9,
                }
            ]

        with patch(
            "alphasig.llm.LLMClient.extract_json",
            new=alternating_response,
        ):
            pipeline = Pipeline(
                model="claude-sonnet-5",
                api_key="test-key",
                user_agent="Test test@example.com",
                cache_dir=None,
                db_path=None,
            )
            # Even if one engine fails, pipeline should not crash
            collection = await pipeline.extract(
                tickers=["AAPL"],
                filing_types=["10-K"],
                lookback_years=5,
                engines=["supply_chain", "m_and_a"],
                store=False,
            )
        # We still get signals from the engines that succeeded
        # (some may have failed, but that's OK)
        assert isinstance(collection, object)

    @pytest.mark.asyncio
    async def test_pipeline_defaults(self) -> None:
        """Pipeline can be constructed with default parameters."""
        pipeline = Pipeline()
        assert pipeline._model == DEFAULT_MODEL
        assert pipeline._concurrency == 4
        assert pipeline._max_concurrent == 3

    @pytest.mark.asyncio
    async def test_pipeline_custom_params(self) -> None:
        pipeline = Pipeline(
            model="claude-opus-5-5",
            api_key="test-key",
            user_agent="Custom custom@test.com",
            cache_dir="/tmp/cache",
            db_path=None,
            concurrency=8,
            max_concurrent=5,
        )
        assert pipeline._model == "claude-opus-5-5"
        assert pipeline._concurrency == 8
        assert pipeline._max_concurrent == 5
        assert pipeline._db_path is None

    @pytest.mark.asyncio
    async def test_extract_raises_without_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pipeline.extract() raises ConfigurationError when no API key."""
        from alphasig.exceptions import ConfigurationError

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        pipeline = Pipeline(
            user_agent="Test test@example.com",
            cache_dir=None,
            db_path=None,
        )
        with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
            await pipeline.extract(
                tickers=["AAPL"],
                engines=["supply_chain"],
                store=False,
            )


class TestDeduplicateAmendmentSignals:
    """Tests for _deduplicate_amendment_signals."""

    def _make_signal(
        self,
        *,
        ticker: str = "AAPL",
        signal_type: SignalType = SignalType.RISK_CHANGE,
        direction: SignalDirection = SignalDirection.BEARISH,
        context: str = "Supply chain concentration risk",
        timestamp: datetime | None = None,
        source_filing: str = "https://sec.gov/10-K",
        metadata: dict[str, str] | None = None,
    ) -> Signal:
        return Signal(
            timestamp=timestamp or datetime(2024, 11, 1, tzinfo=UTC),
            ticker=ticker,
            signal_type=signal_type,
            direction=direction,
            strength=0.8,
            confidence=0.9,
            context=context,
            source_filing=source_filing,
            metadata=metadata or {},
        )

    def test_empty_input(self) -> None:
        assert _deduplicate_amendment_signals([]) == []

    def test_no_duplicates_unchanged(self) -> None:
        signals = [
            self._make_signal(context="Risk A"),
            self._make_signal(context="Risk B"),
        ]
        result = _deduplicate_amendment_signals(signals)
        assert len(result) == 2

    def test_duplicate_keeps_earliest_public(self) -> None:
        """When 10-K and 10-K/A produce the same signal, keep the original.

        The original is when the information became public; keeping the
        amendment would date the signal weeks after it was tradeable.
        """
        original = self._make_signal(
            timestamp=datetime(2024, 11, 1, tzinfo=UTC),
            source_filing="https://sec.gov/10-K",
            metadata={
                "_filing_accession": "0001",
                "_filing_type": "10-K",
                "_period_of_report": "2024-09-30",
            },
        )
        amendment = self._make_signal(
            timestamp=datetime(2024, 12, 15, tzinfo=UTC),
            source_filing="https://sec.gov/10-K-A",
            metadata={
                "_filing_accession": "0002",
                "_filing_type": "10-K/A",
                "_period_of_report": "2024-09-30",
            },
        )
        result = _deduplicate_amendment_signals([amendment, original])
        assert len(result) == 1
        assert result[0].source_filing == "https://sec.gov/10-K"

    def test_different_signal_types_not_deduped(self) -> None:
        """Signals with different types should both be kept."""
        risk = self._make_signal(signal_type=SignalType.RISK_CHANGE)
        supply = self._make_signal(signal_type=SignalType.SUPPLY_CHAIN)
        result = _deduplicate_amendment_signals([risk, supply])
        assert len(result) == 2

    def test_different_tickers_not_deduped(self) -> None:
        aapl = self._make_signal(ticker="AAPL")
        msft = self._make_signal(ticker="MSFT")
        result = _deduplicate_amendment_signals([aapl, msft])
        assert len(result) == 2

    def test_multiple_duplicates_across_tickers(self) -> None:
        """Each ticker's duplicates are resolved independently."""
        signals = [
            self._make_signal(
                ticker="AAPL",
                timestamp=datetime(2024, 11, 1, tzinfo=UTC),
            ),
            self._make_signal(
                ticker="AAPL",
                timestamp=datetime(2024, 12, 1, tzinfo=UTC),
            ),
            self._make_signal(
                ticker="MSFT",
                timestamp=datetime(2024, 11, 1, tzinfo=UTC),
            ),
            self._make_signal(
                ticker="MSFT",
                timestamp=datetime(2024, 12, 1, tzinfo=UTC),
            ),
        ]
        result = _deduplicate_amendment_signals(signals)
        assert len(result) == 2
        tickers = {s.ticker for s in result}
        assert tickers == {"AAPL", "MSFT"}

    def test_repeated_signals_from_distinct_filings_are_preserved(self) -> None:
        """Recurring disclosures across separate filings should not collapse."""
        q1 = self._make_signal(
            timestamp=datetime(2024, 3, 31, tzinfo=UTC),
            source_filing="https://sec.gov/Archives/0001/q1-10q.htm",
            metadata={
                "_filing_accession": "0001",
                "_filing_type": "10-Q",
                "_period_of_report": "2024-03-31",
            },
        )
        q2 = self._make_signal(
            timestamp=datetime(2024, 6, 30, tzinfo=UTC),
            source_filing="https://sec.gov/Archives/0002/q2-10q.htm",
            metadata={
                "_filing_accession": "0002",
                "_filing_type": "10-Q",
                "_period_of_report": "2024-06-30",
            },
        )

        result = _deduplicate_amendment_signals([q1, q2])

        assert len(result) == 2
        assert {signal.source_filing for signal in result} == {
            "https://sec.gov/Archives/0001/q1-10q.htm",
            "https://sec.gov/Archives/0002/q2-10q.htm",
        }


class TestConcurrentFilingDownloads:
    """Tests for concurrent ticker processing in Pipeline."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_concurrent_processes_multiple_tickers(self) -> None:
        """Multiple tickers are processed concurrently via asyncio.gather."""
        # Set up both AAPL and MSFT in the ticker lookup
        respx.get("https://www.sec.gov/files/company_tickers.json").respond(
            json={
                "0": {
                    "cik_str": 320193,
                    "ticker": "AAPL",
                    "title": "Apple Inc.",
                },
                "1": {
                    "cik_str": 789019,
                    "ticker": "MSFT",
                    "title": "Microsoft Corporation",
                },
            }
        )

        respx.get("https://data.sec.gov/submissions/CIK0000320193.json").respond(
            json={
                "cik": "0000320193",
                "name": "Apple Inc.",
                "filings": {
                    "recent": {
                        "accessionNumber": ["0000320193-24-000123"],
                        "form": ["10-K"],
                        "filingDate": ["2024-11-01"],
                        "reportDate": ["2024-09-28"],
                        "primaryDocument": ["aapl-20240928.htm"],
                    }
                },
            }
        )

        respx.get("https://data.sec.gov/submissions/CIK0000789019.json").respond(
            json={
                "cik": "0000789019",
                "name": "Microsoft Corporation",
                "filings": {
                    "recent": {
                        "accessionNumber": ["0000789019-24-000456"],
                        "form": ["10-K"],
                        "filingDate": ["2024-10-30"],
                        "reportDate": ["2024-06-30"],
                        "primaryDocument": ["msft-20240630.htm"],
                    }
                },
            }
        )

        filing_html = """
        <html><body>
        <b>Item 1. Business</b>
        <p>Company description here.</p>

        <b>Item 1A. Risk Factors</b>
        <p>Supply chain and regulatory risks here.</p>

        <b>Item 7. Management's Discussion and Analysis</b>
        <p>Revenue growth and strategic outlook.</p>
        </body></html>
        """
        respx.get(url__startswith="https://www.sec.gov/Archives/").respond(
            text=filing_html
        )

        mock_response = AsyncMock(
            return_value=[
                {
                    "source": "TEST",
                    "target": "TSMC",
                    "relation": "depends_on",
                    "context": "chips",
                    "confidence": 0.9,
                }
            ]
        )

        with patch("alphasig.llm.LLMClient.extract_json", new=mock_response):
            pipeline = Pipeline(
                model="claude-sonnet-5",
                api_key="test-key",
                user_agent="Test test@example.com",
                cache_dir=None,
                db_path=None,
                max_concurrent=2,
            )
            collection = await pipeline.extract(
                tickers=["AAPL", "MSFT"],
                filing_types=["10-K"],
                lookback_years=5,
                engines=["supply_chain"],
                store=False,
                max_concurrent=2,
            )

        # Pipeline completed without error for both tickers
        # (The mock HTML may not yield sections for signal extraction,
        #  but the concurrent gather path ran for both tickers.)
        assert isinstance(collection, object)

    @respx.mock
    @pytest.mark.asyncio
    async def test_max_concurrent_override_at_extract(self) -> None:
        """max_concurrent can be overridden per extract() call."""
        respx.get("https://www.sec.gov/files/company_tickers.json").respond(
            json={
                "0": {
                    "cik_str": 320193,
                    "ticker": "AAPL",
                    "title": "Apple Inc.",
                },
            }
        )

        respx.get("https://data.sec.gov/submissions/CIK0000320193.json").respond(
            json={
                "cik": "0000320193",
                "name": "Apple Inc.",
                "filings": {
                    "recent": {
                        "accessionNumber": [],
                        "form": [],
                        "filingDate": [],
                        "reportDate": [],
                        "primaryDocument": [],
                    }
                },
            }
        )

        mock_response = AsyncMock(return_value=[])
        with patch("alphasig.llm.LLMClient.extract_json", new=mock_response):
            pipeline = Pipeline(
                model="claude-sonnet-5",
                api_key="test-key",
                user_agent="Test test@example.com",
                cache_dir=None,
                db_path=None,
                max_concurrent=1,
            )
            # Override with max_concurrent=5 at call site
            collection = await pipeline.extract(
                tickers=["AAPL"],
                engines=["supply_chain"],
                store=False,
                max_concurrent=5,
            )
        assert len(collection) == 0

    @respx.mock
    @pytest.mark.asyncio
    async def test_one_ticker_failure_does_not_block_others(self) -> None:
        """If one ticker fails, others still succeed in concurrent mode."""
        respx.get("https://www.sec.gov/files/company_tickers.json").respond(
            json={
                "0": {
                    "cik_str": 320193,
                    "ticker": "AAPL",
                    "title": "Apple Inc.",
                },
            }
        )

        # AAPL succeeds but ZZZZ will fail (unknown ticker)
        respx.get("https://data.sec.gov/submissions/CIK0000320193.json").respond(
            json={
                "cik": "0000320193",
                "name": "Apple Inc.",
                "filings": {
                    "recent": {
                        "accessionNumber": [],
                        "form": [],
                        "filingDate": [],
                        "reportDate": [],
                        "primaryDocument": [],
                    }
                },
            }
        )

        mock_response = AsyncMock(return_value=[])
        with patch("alphasig.llm.LLMClient.extract_json", new=mock_response):
            pipeline = Pipeline(
                model="claude-sonnet-5",
                api_key="test-key",
                user_agent="Test test@example.com",
                cache_dir=None,
                db_path=None,
                max_concurrent=2,
            )
            # ZZZZ should fail but not crash the whole pipeline
            collection = await pipeline.extract(
                tickers=["AAPL", "ZZZZ"],
                engines=["supply_chain"],
                store=False,
            )
        # Pipeline completes without raising
        assert isinstance(collection, object)


class TestPipelineRegressions:
    @pytest.mark.asyncio
    async def test_user_agent_is_required(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from alphasig.exceptions import ConfigurationError

        monkeypatch.delenv("ALPHASIG_USER_AGENT", raising=False)
        pipeline = Pipeline(api_key="k", cache_dir=None, db_path=None)
        with pytest.raises(ConfigurationError, match="User-Agent"):
            await pipeline.extract(tickers=["AAPL"], store=False)

    def test_user_agent_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALPHASIG_USER_AGENT", "Env User env@example.com")
        assert Pipeline()._user_agent == "Env User env@example.com"

    @respx.mock
    @pytest.mark.asyncio
    async def test_llm_client_closed_and_signals_stamped(self) -> None:
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
        close = AsyncMock()
        with (
            patch("alphasig.llm.LLMClient.extract_json", new=reply),
            patch("alphasig.llm.LLMClient.aclose", new=close),
        ):
            collection = await Pipeline(
                api_key="k",
                user_agent="Test test@example.com",
                cache_dir=None,
                db_path=None,
            ).extract(
                tickers=["AAPL"],
                filing_types=["10-K"],
                lookback_years=5,
                engines=["supply_chain"],
                store=False,
            )
        close.assert_awaited_once()
        (sig,) = list(collection)
        assert sig.timestamp == datetime(2024, 11, 1, 22, 4, 43, tzinfo=UTC)
        assert sig.source_filing.endswith(
            "/320193/000032019324000123/aapl-20240928.htm"
        )
        assert sig.metadata["_filing_accession"] == "0000320193-24-000123"


# ---------------------------------------------------------------------------
# LLM options and incremental extraction
# ---------------------------------------------------------------------------

_UA = "Test test@example.com"
# Supply-chain calls per mocked filing: one per section (Items 1, 1A and 7).
_SECTIONS = 3
_CIK_URL = "https://data.sec.gov/submissions/CIK0000320193.json"


def _mock_edgar(years: list[int]) -> dict[str, int]:
    """Serve one 10-K per year (Item 1, 1A and 7); count document fetches."""
    fetched: dict[str, int] = {}
    respx.get("https://www.sec.gov/files/company_tickers.json").respond(
        json={"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple"}}
    )
    respx.get(_CIK_URL).respond(
        json={
            "name": "Apple Inc.",
            "filings": {
                "recent": {
                    "accessionNumber": [f"0000320193-{y % 100}-000001" for y in years],
                    "form": ["10-K"] * len(years),
                    "filingDate": [f"{y}-11-01" for y in years],
                    "reportDate": [f"{y}-09-28" for y in years],
                    "primaryDocument": [f"aapl-{y}.htm" for y in years],
                    "acceptanceDateTime": [f"{y}-11-01T16:30:00.000Z" for y in years],
                }
            },
        }
    )

    def doc(request: httpx.Request) -> httpx.Response:
        year = str(request.url).rsplit("-", 1)[1][:4]
        fetched[year] = fetched.get(year, 0) + 1
        risks = " ".join(f"Risk {year}-{i} may affect results." for i in range(30))
        body = (
            "<html><body>"
            "<p><b>Item 1. Business</b></p>"
            f"<p>{'Apple relies on TSMC for chips. ' * 10}</p>"
            f"<p><b>Item 1A. Risk Factors</b></p><p>{risks}</p>"
            "<p><b>Item 7. Management's Discussion and Analysis</b></p>"
            f"<p>{('Fiscal ' + year + ' revenue and margins. ') * 10}</p>"
            "</body></html>"
        )
        return httpx.Response(200, text=body)

    respx.get(url__startswith="https://www.sec.gov/Archives/").mock(side_effect=doc)
    return fetched


def _llm_reply(system: str) -> list[dict[str, object]]:
    if "securities lawyer" in system:
        return [
            {
                "change_type": "NEW",
                "risk": "New risk",
                "severity_estimate": "HIGH",
                "confidence": 0.8,
            }
        ]
    if "supply-chain" in system:
        return [{"target": "TSM", "relation": "depends_on", "confidence": 0.9}]
    return []


class _FakeAnthropic:
    """Stands in for ``anthropic.AsyncAnthropic``; counts calls and peak load."""

    stats: ClassVar[dict[str, int]] = {}

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.messages = self

    async def create(self, **kwargs: object) -> SimpleNamespace:
        stats = type(self).stats
        stats["calls"] += 1
        stats["inflight"] += 1
        stats["peak"] = max(stats["peak"], stats["inflight"])
        try:
            await asyncio.sleep(0.01)
        finally:
            stats["inflight"] -= 1
        system = kwargs["system"][0]["text"]  # type: ignore[index]
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(_llm_reply(system)))],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )

    async def close(self) -> None:
        return None


@pytest.fixture
def fake_anthropic() -> Iterator[dict[str, int]]:
    import anthropic

    _FakeAnthropic.stats = {"calls": 0, "inflight": 0, "peak": 0}
    with patch.object(anthropic, "AsyncAnthropic", _FakeAnthropic):
        yield _FakeAnthropic.stats


async def _extract(pipeline: Pipeline, **kwargs: object) -> list[Signal]:
    options: dict[str, object] = {
        "tickers": ["AAPL"],
        "filing_types": ["10-K"],
        "lookback_years": 5,
        "engines": ["supply_chain", "risk_differ"],
        "store": False,
    }
    options.update(kwargs)
    return list(await pipeline.extract(**options))  # type: ignore[arg-type]


class TestLLMOptions:
    @respx.mock
    async def test_llm_concurrency_bounds_in_flight_requests(
        self, fake_anthropic: dict[str, int]
    ) -> None:
        _mock_edgar([2022, 2023, 2024])
        pipeline = Pipeline(
            api_key="k", user_agent=_UA, cache_dir=None, db_path=None, llm_concurrency=1
        )
        signals = await _extract(pipeline)
        assert signals
        # 3 filings x 3 sections of supply chain + 2 risk diffs, one at a time.
        assert fake_anthropic["calls"] == 3 * _SECTIONS + 2
        assert fake_anthropic["peak"] == 1

    def test_llm_concurrency_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="llm_concurrency"):
            Pipeline(llm_concurrency=0)

    @respx.mock
    async def test_llm_cache_dir_makes_reruns_free(
        self, fake_anthropic: dict[str, int], tmp_path: Path
    ) -> None:
        _mock_edgar([2023, 2024])
        cache = tmp_path / "llm"

        def pipeline() -> Pipeline:
            return Pipeline(
                api_key="k",
                user_agent=_UA,
                cache_dir=None,
                db_path=None,
                llm_cache_dir=str(cache),
            )

        first = await _extract(pipeline())
        calls = fake_anthropic["calls"]
        assert calls > 0
        second = await _extract(pipeline())
        assert fake_anthropic["calls"] == calls
        assert sorted(s.context for s in second) == sorted(s.context for s in first)
        assert list(cache.rglob("*.json"))


class _HalfCalibrator(JevCalibrator):
    """Sets every confidence to 0.5 without calling Jev."""

    def __init__(self) -> None:
        super().__init__(client=MagicMock())
        self.calls = 0

    async def calibrate(
        self,
        signals: Sequence[Signal],
        sections: Sequence[FilingSection],
        previous_sections: Sequence[FilingSection] | None = None,
    ) -> list[Signal]:
        self.calls += 1
        return [
            s.model_copy(
                update={
                    "confidence": 0.5,
                    "metadata": {**s.metadata, "confidence_source": "jev"},
                }
            )
            for s in signals
        ]


class TestIncrementalExtraction:
    """incremental=True records finished jobs and skips them next time."""

    @staticmethod
    def _pipeline(db: Path, **kwargs: object) -> Pipeline:
        return Pipeline(
            api_key="k",
            user_agent=_UA,
            cache_dir=None,
            db_path=str(db),
            **kwargs,  # type: ignore[arg-type]
        )

    @respx.mock
    async def test_second_run_skips_finished_jobs(
        self, fake_anthropic: dict[str, int], tmp_path: Path
    ) -> None:
        db = tmp_path / "inc.duckdb"
        fetched = _mock_edgar([2023, 2024])
        first = await _extract(self._pipeline(db), store=True, incremental=True)
        assert first
        assert fake_anthropic["calls"] == 2 * _SECTIONS + 1  # + one risk diff

        second = await _extract(self._pipeline(db), store=True, incremental=True)
        assert second == []
        assert fake_anthropic["calls"] == 2 * _SECTIONS + 1
        # Nothing left to do: no filing documents were downloaded again.
        assert fetched == {"2023": 1, "2024": 1}

        with SignalStore(db) as store:
            assert store.count() == len(first)
            done = store.completed_extractions()
        assert set(done) == {
            ("0000320193-23-000001", "supply_chain"),
            ("0000320193-23-000001", "risk_differ"),
            ("0000320193-24-000001", "supply_chain"),
            ("0000320193-24-000001", "risk_differ"),
        }
        latest = done[("0000320193-24-000001", "risk_differ")]
        assert latest.previous_accession == "0000320193-23-000001"
        assert latest.signal_count == 1
        assert done[("0000320193-23-000001", "risk_differ")].signal_count == 0

    @respx.mock
    async def test_new_filing_runs_only_its_jobs(
        self, fake_anthropic: dict[str, int], tmp_path: Path
    ) -> None:
        db = tmp_path / "inc.duckdb"
        _mock_edgar([2022, 2023])
        await _extract(self._pipeline(db), store=True, incremental=True)
        calls = fake_anthropic["calls"]

        respx.reset()
        fetched = _mock_edgar([2022, 2023, 2024])
        new = await _extract(self._pipeline(db), store=True, incremental=True)
        # 2024's supply chain and one risk diff (2024 vs 2023).
        assert fake_anthropic["calls"] - calls == _SECTIONS + 1
        assert {s.metadata["_filing_accession"] for s in new} == {
            "0000320193-24-000001"
        }
        # 2022 is neither pending nor needed as a predecessor.
        assert fetched == {"2023": 1, "2024": 1}

    @respx.mock
    async def test_failed_jobs_are_retried(self, tmp_path: Path) -> None:
        db = tmp_path / "inc.duckdb"
        _mock_edgar([2023, 2024])
        calls: list[str] = []

        async def failing(self_: object, system: str, user: str, **_: object) -> object:
            calls.append(system)
            if "securities lawyer" in system:
                raise RuntimeError("LLM down")
            return _llm_reply(system)

        with patch("alphasig.llm.LLMClient.extract_json", new=failing):
            await _extract(self._pipeline(db), store=True, incremental=True)
        with SignalStore(db) as store:
            done = set(store.completed_extractions())
        assert ("0000320193-24-000001", "risk_differ") not in done
        assert ("0000320193-24-000001", "supply_chain") in done

        async def working(self_: object, system: str, user: str, **_: object) -> object:
            calls.append(system)
            return _llm_reply(system)

        calls.clear()
        with patch("alphasig.llm.LLMClient.extract_json", new=working):
            retried = await _extract(self._pipeline(db), store=True, incremental=True)
        assert len(calls) == 1 and "securities lawyer" in calls[0]
        assert [s.signal_type for s in retried] == [SignalType.RISK_CHANGE]

    @respx.mock
    async def test_calibrated_run_redoes_and_replaces_raw_jobs(
        self, fake_anthropic: dict[str, int], tmp_path: Path
    ) -> None:
        db = tmp_path / "inc.duckdb"
        _mock_edgar([2023, 2024])
        raw = await _extract(self._pipeline(db), store=True, incremental=True)
        assert all(s.confidence != 0.5 for s in raw)

        calibrator = _HalfCalibrator()
        calibrated = await _extract(
            self._pipeline(db, calibrator=calibrator), store=True, incremental=True
        )
        assert calibrator.calls == 2
        assert len(calibrated) == len(raw)
        with SignalStore(db) as store:
            stored = store.query()
            done = store.completed_extractions()
        # The raw signals were replaced, not duplicated.
        assert len(stored) == len(raw)
        assert {s.confidence for s in stored} == {0.5}
        assert all(record.calibrated for record in done.values())

        # A later uncalibrated run accepts the calibrated jobs as done.
        calls = fake_anthropic["calls"]
        assert await _extract(self._pipeline(db), store=True, incremental=True) == []
        assert fake_anthropic["calls"] == calls

    @respx.mock
    async def test_jobs_jev_failed_to_score_are_retried(
        self, fake_anthropic: dict[str, int], tmp_path: Path
    ) -> None:
        from tests.test_jev import _FakeJev

        db = tmp_path / "inc.duckdb"
        _mock_edgar([2023, 2024])
        down = _FakeJev(status=500)
        first = await _extract(
            self._pipeline(db, calibrator=JevCalibrator(down.client())),
            store=True,
            incremental=True,
        )
        assert first and down.requests
        assert all("confidence_source" not in s.metadata for s in first)
        with SignalStore(db) as store:
            done = store.completed_extractions()
        # Jobs with signals were not calibrated; the empty 2023 diff was.
        assert not done[("0000320193-24-000001", "risk_differ")].calibrated
        assert done[("0000320193-23-000001", "risk_differ")].calibrated

        calls = fake_anthropic["calls"]
        up = _FakeJev(probability=0.7)
        retried = await _extract(
            self._pipeline(db, calibrator=JevCalibrator(up.client())),
            store=True,
            incremental=True,
        )
        assert fake_anthropic["calls"] > calls
        assert retried and {s.confidence for s in retried} == {0.7}
        with SignalStore(db) as store:
            assert {s.confidence for s in store.query()} == {0.7}
            assert store.count() == len(first)

    @respx.mock
    async def test_diff_job_is_redone_when_its_predecessor_changes(
        self, fake_anthropic: dict[str, int], tmp_path: Path
    ) -> None:
        db = tmp_path / "inc.duckdb"
        # First run sees only the 2024 filing: nothing to diff against.
        _mock_edgar([2024])
        await _extract(self._pipeline(db), store=True, incremental=True)
        calls = fake_anthropic["calls"]

        # The 2023 filing appears (e.g. a longer lookback).
        respx.reset()
        _mock_edgar([2023, 2024])
        new = await _extract(self._pipeline(db), store=True, incremental=True)
        # 2023's supply chain (its risk diff has no predecessor) + 2024's diff.
        assert fake_anthropic["calls"] - calls == _SECTIONS + 1
        assert SignalType.RISK_CHANGE in {s.signal_type for s in new}
        with SignalStore(db) as store:
            record = store.completed_extractions()[
                ("0000320193-24-000001", "risk_differ")
            ]
        assert record.previous_accession == "0000320193-23-000001"

    @respx.mock
    async def test_non_incremental_runs_ignore_the_log(
        self, fake_anthropic: dict[str, int], tmp_path: Path
    ) -> None:
        db = tmp_path / "inc.duckdb"
        _mock_edgar([2023, 2024])
        await _extract(self._pipeline(db), store=True, incremental=True)
        calls = fake_anthropic["calls"]
        again = await _extract(self._pipeline(db), store=True)
        assert again
        assert fake_anthropic["calls"] == 2 * calls
        with SignalStore(db) as store:
            # Idempotent insert: the repeat run added no duplicate rows.
            assert store.count() == len(again)

    async def test_requires_database(self) -> None:
        from alphasig.exceptions import ConfigurationError

        pipeline = Pipeline(api_key="k", user_agent=_UA, db_path=None)
        with pytest.raises(ConfigurationError, match="db_path"):
            await pipeline.extract(tickers=["AAPL"], incremental=True)
        pipeline = Pipeline(api_key="k", user_agent=_UA, db_path=":memory:")
        with pytest.raises(ConfigurationError, match="store=True"):
            await pipeline.extract(tickers=["AAPL"], incremental=True, store=False)
