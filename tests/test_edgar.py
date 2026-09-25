"""Tests for alphasig.edgar -- EDGAR API client."""

from __future__ import annotations

import httpx
import pytest
import respx

from alphasig.edgar import EdgarClient
from alphasig.exceptions import EdgarNotFoundError
from alphasig.models import FilingType

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


class TestEdgarFairAccessAndCorrectness:
    """Regression tests for SEC fair-access and filing-metadata handling."""

    @pytest.mark.asyncio
    async def test_rate_limiter_spaces_concurrent_requests(self) -> None:
        import asyncio
        import time

        from alphasig.edgar import _RateLimiter

        limiter = _RateLimiter(max_per_second=20)
        stamps: list[float] = []

        async def hit() -> None:
            await limiter.acquire()
            stamps.append(time.monotonic())

        start = time.monotonic()
        await asyncio.gather(*(hit() for _ in range(6)))
        # No burst: request i cannot start before its own 50 ms slot.
        for i, stamp in enumerate(sorted(stamps)):
            assert stamp - start >= i * 0.05 - 0.005

    @respx.mock
    @pytest.mark.asyncio
    async def test_retry_after_header_is_honoured(
        self, mock_company_tickers: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        waits: list[float] = []

        async def record_sleep(seconds: float) -> None:
            waits.append(seconds)

        monkeypatch.setattr(EdgarClient._get.retry, "sleep", record_sleep)
        respx.get("https://www.sec.gov/files/company_tickers.json").mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": "7"}),
                httpx.Response(200, json=mock_company_tickers),
            ]
        )
        async with EdgarClient(
            user_agent="Test test@example.com", cache_dir=None
        ) as client:
            assert await client.resolve_cik("AAPL") == "0000320193"
        assert waits == [7.0]

    @respx.mock
    @pytest.mark.asyncio
    async def test_403_is_not_retried_and_explains_policy(self) -> None:
        from alphasig.exceptions import EdgarError

        route = respx.get("https://www.sec.gov/files/company_tickers.json").respond(403)
        async with EdgarClient(
            user_agent="Test test@example.com", cache_dir=None
        ) as client:
            with pytest.raises(EdgarError, match="User-Agent"):
                await client.resolve_cik("AAPL")
        assert route.call_count == 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_concurrent_resolve_cik_downloads_ticker_map_once(
        self, mock_company_tickers: dict
    ) -> None:
        import asyncio

        route = respx.get("https://www.sec.gov/files/company_tickers.json").respond(
            json=mock_company_tickers
        )
        async with EdgarClient(
            user_agent="Test test@example.com", cache_dir=None
        ) as client:
            ciks = await asyncio.gather(
                client.resolve_cik("AAPL"), client.resolve_cik("MSFT")
            )
        assert ciks == ["0000320193", "0000789019"]
        assert route.call_count == 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_acceptance_time_is_eastern_and_url_uses_unpadded_cik(
        self, mock_company_tickers: dict, mock_submissions: dict
    ) -> None:
        from datetime import UTC, datetime

        recent = mock_submissions["filings"]["recent"]
        recent["acceptanceDateTime"] = [
            "2024-11-01T18:04:43.000Z",
            "2024-05-02T18:03:07.000Z",
            "2023-11-02T18:08:27.000Z",
        ]
        respx.get("https://www.sec.gov/files/company_tickers.json").respond(
            json=mock_company_tickers
        )
        respx.get("https://data.sec.gov/submissions/CIK0000320193.json").respond(
            json=mock_submissions
        )
        async with EdgarClient(
            user_agent="Test test@example.com", cache_dir=None
        ) as client:
            filings = await client.get_filings("AAPL", lookback_years=5)
        latest = filings[-1]
        # EDGAR's "Z" suffix is misleading: 18:04 is Eastern (EDT on Nov 1).
        assert latest.available_at == datetime(2024, 11, 1, 22, 4, 43, tzinfo=UTC)
        assert latest.url == (
            "https://www.sec.gov/Archives/edgar/data/320193/"
            "000032019324000123/aapl-20240928.htm"
        )

    def test_available_at_without_acceptance_uses_filing_cutoff(self) -> None:
        from datetime import UTC, date, datetime

        from alphasig.models import public_availability

        # 17:30 EDT on the filing date, never midnight (which precedes release).
        assert public_availability(date(2024, 7, 1), None) == datetime(
            2024, 7, 1, 21, 30, tzinfo=UTC
        )

    @respx.mock
    @pytest.mark.asyncio
    async def test_older_filings_pages_are_followed(
        self, mock_company_tickers: dict, mock_submissions: dict
    ) -> None:
        from datetime import date, timedelta

        recent_year = date.today().year
        mock_submissions["filings"]["files"] = [
            {
                "name": "CIK0000320193-submissions-001.json",
                "filingFrom": f"{recent_year - 2}-01-01",
                "filingTo": (date.today() - timedelta(days=200)).isoformat(),
            },
            {
                "name": "CIK0000320193-submissions-002.json",
                "filingFrom": "1994-01-01",
                "filingTo": "2001-12-31",
            },
        ]
        old_date = (date.today() - timedelta(days=300)).isoformat()
        respx.get("https://www.sec.gov/files/company_tickers.json").respond(
            json=mock_company_tickers
        )
        respx.get("https://data.sec.gov/submissions/CIK0000320193.json").respond(
            json=mock_submissions
        )
        page = respx.get(
            "https://data.sec.gov/submissions/CIK0000320193-submissions-001.json"
        ).respond(
            json={
                "accessionNumber": ["0000320193-99-000001"],
                "form": ["10-K"],
                "filingDate": [old_date],
                "reportDate": [old_date],
                "primaryDocument": ["old.htm"],
            }
        )
        ancient = respx.get(
            "https://data.sec.gov/submissions/CIK0000320193-submissions-002.json"
        )
        async with EdgarClient(
            user_agent="Test test@example.com", cache_dir=None
        ) as client:
            filings = await client.get_filings("AAPL", lookback_years=5)
        assert page.called
        assert not ancient.called  # entirely before the lookback window
        assert "0000320193-99-000001" in {f.accession_number for f in filings}

    @respx.mock
    @pytest.mark.asyncio
    async def test_amendments_are_not_mixed_into_originals(
        self, mock_company_tickers: dict, mock_submissions: dict
    ) -> None:
        recent = mock_submissions["filings"]["recent"]
        recent["accessionNumber"].append("0000320193-24-000200")
        recent["form"].append("10-K/A")
        recent["filingDate"].append("2024-12-01")
        recent["reportDate"].append("2024-09-28")
        recent["primaryDocument"].append("aapl-10ka.htm")
        respx.get("https://www.sec.gov/files/company_tickers.json").respond(
            json=mock_company_tickers
        )
        respx.get("https://data.sec.gov/submissions/CIK0000320193.json").respond(
            json=mock_submissions
        )
        async with EdgarClient(
            user_agent="Test test@example.com", cache_dir=None
        ) as client:
            filings = await client.get_filings("AAPL", ["10-K"], lookback_years=5)
        assert [f.accession_number for f in filings] == [
            "0000320193-23-000100",
            "0000320193-24-000123",
        ]

    @respx.mock
    @pytest.mark.asyncio
    async def test_cache_write_leaves_no_partial_file(
        self, mock_company_tickers: dict, mock_submissions: dict, tmp_path
    ) -> None:
        respx.get("https://www.sec.gov/files/company_tickers.json").respond(
            json=mock_company_tickers
        )
        respx.get("https://data.sec.gov/submissions/CIK0000320193.json").respond(
            json=mock_submissions
        )
        respx.get(url__startswith="https://www.sec.gov/Archives/").respond(
            text="<html>body</html>"
        )
        async with EdgarClient(
            user_agent="Test test@example.com", cache_dir=str(tmp_path)
        ) as client:
            filings = await client.get_filings("AAPL", lookback_years=5)
            await client.fetch_filing_html(filings[0])
        assert [p.suffix for p in tmp_path.iterdir()] == [".html"]
