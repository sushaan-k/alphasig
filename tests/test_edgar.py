"""Tests for sigint.edgar -- EDGAR API client."""

from __future__ import annotations

import httpx
import pytest
import respx

from sigint.edgar import EdgarClient
from sigint.exceptions import EdgarNotFoundError
from sigint.models import FilingType

# -- Fixtures ----------------------------------------------------------------


@pytest.fixture
def mock_company_tickers() -> dict:
    """Mock response for company_tickers.json."""
    return {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
    }


@pytest.fixture
def mock_submissions() -> dict:
    """Mock response for EDGAR submissions API."""
    return {
        "cik": "0000320193",
        "name": "Apple Inc.",
        "filings": {
            "recent": {
                "accessionNumber": [
                    "0000320193-24-000123",
                    "0000320193-24-000050",
                    "0000320193-23-000100",
                ],
                "form": ["10-K", "10-Q", "10-K"],
                "filingDate": ["2024-11-01", "2024-05-03", "2023-11-03"],
                "reportDate": ["2024-09-28", "2024-03-30", "2023-09-30"],
                "primaryDocument": [
                    "aapl-20240928.htm",
                    "aapl-20240330.htm",
                    "aapl-20230930.htm",
                ],
            }
        },
    }


# -- Tests -------------------------------------------------------------------


class TestEdgarClient:
    """Tests for the EdgarClient."""

    def test_requires_email_in_user_agent(self) -> None:
        with pytest.raises(ValueError, match="contact email"):
            EdgarClient(user_agent="no-email-here")

    @respx.mock
    @pytest.mark.asyncio
    async def test_resolve_cik(self, mock_company_tickers: dict) -> None:
        respx.get("https://www.sec.gov/files/company_tickers.json").respond(
            json=mock_company_tickers
        )
        async with EdgarClient(
            user_agent="Test test@example.com", cache_dir=None
        ) as client:
            cik = await client.resolve_cik("AAPL")
            assert cik == "0000320193"

    @respx.mock
    @pytest.mark.asyncio
    async def test_resolve_cik_unknown_ticker(self, mock_company_tickers: dict) -> None:
        respx.get("https://www.sec.gov/files/company_tickers.json").respond(
            json=mock_company_tickers
        )
        async with EdgarClient(
            user_agent="Test test@example.com", cache_dir=None
        ) as client:
            with pytest.raises(EdgarNotFoundError):
                await client.resolve_cik("ZZZZ")

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_filings(
        self,
        mock_company_tickers: dict,
        mock_submissions: dict,
    ) -> None:
        respx.get("https://www.sec.gov/files/company_tickers.json").respond(
            json=mock_company_tickers
        )
        respx.get("https://data.sec.gov/submissions/CIK0000320193.json").respond(
            json=mock_submissions
        )

        async with EdgarClient(
            user_agent="Test test@example.com", cache_dir=None
        ) as client:
            filings = await client.get_filings(
                "AAPL",
                filing_types=["10-K", "10-Q"],
                lookback_years=5,
            )
            assert len(filings) == 3
            assert filings[0].ticker == "AAPL"
            assert filings[0].filing_type in (
                FilingType.TEN_K,
                FilingType.TEN_Q,
            )
            # Sorted by date
            assert filings[0].filed_date <= filings[-1].filed_date

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_filings_filters_by_type(
        self,
        mock_company_tickers: dict,
        mock_submissions: dict,
    ) -> None:
        respx.get("https://www.sec.gov/files/company_tickers.json").respond(
            json=mock_company_tickers
        )
        respx.get("https://data.sec.gov/submissions/CIK0000320193.json").respond(
            json=mock_submissions
        )

        async with EdgarClient(
            user_agent="Test test@example.com", cache_dir=None
        ) as client:
            filings = await client.get_filings(
                "AAPL",
                filing_types=["10-K"],
                lookback_years=5,
            )
            assert all(f.filing_type == FilingType.TEN_K for f in filings)

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_filing_html(
        self,
        mock_company_tickers: dict,
        mock_submissions: dict,
    ) -> None:
        respx.get("https://www.sec.gov/files/company_tickers.json").respond(
            json=mock_company_tickers
        )
        respx.get("https://data.sec.gov/submissions/CIK0000320193.json").respond(
            json=mock_submissions
        )
        respx.get(url__startswith="https://www.sec.gov/Archives/").respond(
            text="<html><body>Filing content</body></html>"
        )

        async with EdgarClient(
            user_agent="Test test@example.com", cache_dir=None
        ) as client:
            filings = await client.get_filings("AAPL", lookback_years=5)
            populated = await client.fetch_filing_html(filings[0])
            assert "Filing content" in populated.raw_html

    @respx.mock
    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, mock_company_tickers: dict) -> None:
        call_count = 0

        def side_effect(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return httpx.Response(429)
            return httpx.Response(200, json=mock_company_tickers)

        respx.get("https://www.sec.gov/files/company_tickers.json").mock(
            side_effect=side_effect
        )
        async with EdgarClient(
            user_agent="Test test@example.com", cache_dir=None
        ) as client:
            cik = await client.resolve_cik("AAPL")
            assert cik == "0000320193"
            assert call_count == 3  # 2 retries + 1 success

    def test_client_must_be_context_manager(self) -> None:
        client = EdgarClient(user_agent="Test test@example.com", cache_dir=None)
        with pytest.raises(Exception, match="context manager"):
            import asyncio

            asyncio.run(client.resolve_cik("AAPL"))

    def test_empty_user_agent_rejected(self) -> None:
        with pytest.raises(ValueError, match="contact email"):
            EdgarClient(user_agent="")

    @respx.mock
    @pytest.mark.asyncio
    async def test_resolve_cik_caches_result(self, mock_company_tickers: dict) -> None:
        route = respx.get("https://www.sec.gov/files/company_tickers.json").respond(
            json=mock_company_tickers
        )
        async with EdgarClient(
            user_agent="Test test@example.com", cache_dir=None
        ) as client:
            cik1 = await client.resolve_cik("AAPL")
            cik2 = await client.resolve_cik("AAPL")
            assert cik1 == cik2
            # The second call should use the in-memory cache, but the HTTP
            # endpoint is only called once for the tickers JSON
            assert route.call_count == 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_resolve_cik_case_insensitive(
        self, mock_company_tickers: dict
    ) -> None:
        respx.get("https://www.sec.gov/files/company_tickers.json").respond(
            json=mock_company_tickers
        )
        async with EdgarClient(
            user_agent="Test test@example.com", cache_dir=None
        ) as client:
            cik = await client.resolve_cik("aapl")
            assert cik == "0000320193"

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_returns_404(self, mock_company_tickers: dict) -> None:
        respx.get("https://www.sec.gov/files/company_tickers.json").respond(
            json=mock_company_tickers
        )
        respx.get("https://data.sec.gov/submissions/CIK0000320193.json").respond(
            status_code=404
        )
        async with EdgarClient(
            user_agent="Test test@example.com", cache_dir=None
        ) as client:
            with pytest.raises(EdgarNotFoundError):
                await client.get_filings("AAPL", lookback_years=5)

    @respx.mock
    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached(
        self, mock_company_tickers: dict, mock_submissions: dict, tmp_path
    ) -> None:
        """Verify that a second fetch of the same filing URL is a cache hit."""
        respx.get("https://www.sec.gov/files/company_tickers.json").respond(
            json=mock_company_tickers
        )
        respx.get("https://data.sec.gov/submissions/CIK0000320193.json").respond(
            json=mock_submissions
        )
        filing_route = respx.get(
            url__startswith="https://www.sec.gov/Archives/"
        ).respond(text="<html>Cached content</html>")

        async with EdgarClient(
            user_agent="Test test@example.com", cache_dir=str(tmp_path)
        ) as client:
            filings = await client.get_filings("AAPL", lookback_years=5)
            # First fetch populates cache
            f1 = await client.fetch_filing_html(filings[0])
            assert "Cached content" in f1.raw_html

            # Second fetch should hit cache (no additional HTTP call)
            call_count_before = filing_route.call_count
            f2 = await client.fetch_filing_html(filings[0])
            assert "Cached content" in f2.raw_html
            assert filing_route.call_count == call_count_before

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_filings_ignores_unsupported_form_types(
        self, mock_company_tickers: dict
    ) -> None:
        """Filings with form types not in FilingType should be skipped."""
        respx.get("https://www.sec.gov/files/company_tickers.json").respond(
            json=mock_company_tickers
        )
        respx.get("https://data.sec.gov/submissions/CIK0000320193.json").respond(
            json={
                "cik": "0000320193",
                "name": "Apple Inc.",
                "filings": {
                    "recent": {
                        "accessionNumber": ["0000320193-24-000999"],
                        "form": ["SC 13G"],
                        "filingDate": ["2024-06-01"],
                        "reportDate": ["2024-06-01"],
                        "primaryDocument": ["filing.htm"],
                    }
                },
            }
        )
        async with EdgarClient(
            user_agent="Test test@example.com", cache_dir=None
        ) as client:
            filings = await client.get_filings(
                "AAPL", filing_types=["SC 13G"], lookback_years=5
            )
            # SC 13G is not in FilingType enum so should be skipped
            assert filings == []

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_filings_respects_lookback(
        self, mock_company_tickers: dict
    ) -> None:
        """Old filings outside the lookback window should be excluded."""
        respx.get("https://www.sec.gov/files/company_tickers.json").respond(
            json=mock_company_tickers
        )
        respx.get("https://data.sec.gov/submissions/CIK0000320193.json").respond(
            json={
                "cik": "0000320193",
                "name": "Apple Inc.",
                "filings": {
                    "recent": {
                        "accessionNumber": ["0000320193-15-000001"],
                        "form": ["10-K"],
                        "filingDate": ["2015-11-01"],
                        "reportDate": ["2015-09-28"],
                        "primaryDocument": ["old.htm"],
                    }
                },
            }
        )
        async with EdgarClient(
            user_agent="Test test@example.com", cache_dir=None
        ) as client:
            filings = await client.get_filings("AAPL", lookback_years=1)
            assert filings == []

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_filings_with_filing_type_enum(
        self, mock_company_tickers: dict, mock_submissions: dict
    ) -> None:
        """Supports passing FilingType enum values directly."""
        respx.get("https://www.sec.gov/files/company_tickers.json").respond(
            json=mock_company_tickers
        )
        respx.get("https://data.sec.gov/submissions/CIK0000320193.json").respond(
            json=mock_submissions
        )
        async with EdgarClient(
            user_agent="Test test@example.com", cache_dir=None
        ) as client:
            filings = await client.get_filings(
                "AAPL", filing_types=[FilingType.TEN_K], lookback_years=5
            )
            assert all(f.filing_type == FilingType.TEN_K for f in filings)

    @respx.mock
    @pytest.mark.asyncio
    async def test_search_full_text(self, mock_company_tickers: dict) -> None:
        respx.get("https://efts.sec.gov/LATEST/search-index").respond(
            json={"hits": {"hits": [{"_id": "1"}, {"_id": "2"}, {"_id": "3"}]}}
        )
        async with EdgarClient(
            user_agent="Test test@example.com", cache_dir=None
        ) as client:
            with pytest.warns(
                UserWarning,
                match="search_full_text\\(\\) is experimental",
            ):
                results = await client.search_full_text("supply chain", limit=2)
            assert len(results) == 2

    def test_cache_dir_creation(self, tmp_path) -> None:
        cache = tmp_path / "sub" / "cache"
        EdgarClient(user_agent="Test test@example.com", cache_dir=str(cache))
        assert cache.exists()
