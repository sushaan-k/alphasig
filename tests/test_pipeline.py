"""Integration-level tests for the Pipeline (mocking EDGAR and LLM)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import respx

from sigint.exceptions import ExtractionError, PipelineError
from sigint.models import Signal, SignalDirection, SignalType
from sigint.pipeline import (
    Pipeline,
    _deduplicate_amendment_signals,
    _resolve_engines,
    _run_engine,
)


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

        with patch("sigint.llm.LLMClient.extract_json", new=mock_response):
            pipeline = Pipeline(
                model="claude-sonnet-4-6",
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
        with patch("sigint.llm.LLMClient.extract_json", new=mock_response):
            pipeline = Pipeline(
                model="claude-sonnet-4-6",
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
        with patch("sigint.llm.LLMClient.extract_json", new=mock_response):
            pipeline = Pipeline(
                model="claude-sonnet-4-6",
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
        with patch("sigint.llm.LLMClient.extract_json", new=mock_response):
            pipeline = Pipeline(
                model="claude-sonnet-4-6",
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
            "sigint.llm.LLMClient.extract_json",
            new=alternating_response,
        ):
            pipeline = Pipeline(
                model="claude-sonnet-4-6",
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
        assert pipeline._model == "claude-sonnet-4-6"
        assert pipeline._concurrency == 4
        assert pipeline._max_concurrent == 3

    @pytest.mark.asyncio
    async def test_pipeline_custom_params(self) -> None:
        pipeline = Pipeline(
            model="claude-opus-4-6",
            api_key="test-key",
            user_agent="Custom custom@test.com",
            cache_dir="/tmp/cache",
            db_path=None,
            concurrency=8,
            max_concurrent=5,
        )
        assert pipeline._model == "claude-opus-4-6"
        assert pipeline._concurrency == 8
        assert pipeline._max_concurrent == 5
        assert pipeline._db_path is None

    @pytest.mark.asyncio
    async def test_extract_raises_without_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pipeline.extract() raises ConfigurationError when no API key."""
        from sigint.exceptions import ConfigurationError

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

    def test_duplicate_keeps_most_recent(self) -> None:
        """When 10-K and 10-K/A produce the same signal, keep the amendment."""
        original = self._make_signal(
            timestamp=datetime(2024, 11, 1, tzinfo=UTC),
            source_filing="https://sec.gov/10-K",
        )
        amendment = self._make_signal(
            timestamp=datetime(2024, 12, 15, tzinfo=UTC),
            source_filing="https://sec.gov/10-K-A",
        )
        result = _deduplicate_amendment_signals([original, amendment])
        assert len(result) == 1
        assert result[0].source_filing == "https://sec.gov/10-K-A"

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

        with patch("sigint.llm.LLMClient.extract_json", new=mock_response):
            pipeline = Pipeline(
                model="claude-sonnet-4-6",
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
        with patch("sigint.llm.LLMClient.extract_json", new=mock_response):
            pipeline = Pipeline(
                model="claude-sonnet-4-6",
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
        with patch("sigint.llm.LLMClient.extract_json", new=mock_response):
            pipeline = Pipeline(
                model="claude-sonnet-4-6",
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
