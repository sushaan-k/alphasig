"""Async EDGAR API client with rate limiting and caching.

SEC EDGAR enforces a limit of 10 requests per second.  This client
respects that constraint using a token-bucket approach built on
:mod:`asyncio` primitives.

Typical usage::

    async with EdgarClient(user_agent="you@example.com") as client:
        filings = await client.get_filings("AAPL", filing_types=["10-K"])
        html = await client.fetch_filing_html(filings[0])
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from sigint.exceptions import (
    EdgarError,
    EdgarNotFoundError,
    EdgarRateLimitError,
)
from sigint.models import Filing, FilingType

logger = structlog.get_logger()

_EDGAR_FULL_TEXT_SEARCH = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions"
_EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

_MAX_RPS = 10  # SEC rate limit


class _RateLimiter:
    """Simple token-bucket rate limiter for EDGAR's 10 req/s policy."""

    def __init__(self, max_per_second: int = _MAX_RPS) -> None:
        self._max = max_per_second
        self._tokens = float(max_per_second)
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._tokens = min(self._max, self._tokens + elapsed * self._max)
            self._last = now
            if self._tokens < 1:
                wait = (1 - self._tokens) / self._max
                await asyncio.sleep(wait)
                # Recalculate tokens after sleeping, then subtract 1
                # for the request being granted.
                now = time.monotonic()
                elapsed = now - self._last
                self._tokens = min(self._max, self._tokens + elapsed * self._max)
                self._last = now
            self._tokens -= 1


class EdgarClient:
    """Async client for SEC EDGAR with rate limiting and disk caching.

    Args:
        user_agent: Required by SEC -- your name and email,
            e.g. ``"Jane Doe jane@example.com"``.
        cache_dir: Local directory for caching raw filings.
            ``None`` disables caching.
    """

    def __init__(
        self,
        user_agent: str,
        cache_dir: str | Path | None = "./edgar_cache",
    ) -> None:
        if not user_agent or "@" not in user_agent:
            raise ValueError(
                "SEC requires a User-Agent with a contact email, "
                "e.g. 'Jane Doe jane@example.com'"
            )
        self._user_agent = user_agent
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._limiter = _RateLimiter()
        self._client: httpx.AsyncClient | None = None
        self._ticker_to_cik: dict[str, str] = {}

    # -- Context manager ------------------------------------------------------

    async def __aenter__(self) -> EdgarClient:
        self._client = httpx.AsyncClient(
            headers={
                "User-Agent": self._user_agent,
                "Accept-Encoding": "gzip, deflate",
            },
            timeout=httpx.Timeout(30.0, connect=10.0),
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    # -- Internal helpers -----------------------------------------------------

    def _assert_open(self) -> httpx.AsyncClient:
        if self._client is None:
            raise EdgarError("EdgarClient must be used as an async context manager")
        return self._client

    @retry(
        retry=retry_if_exception_type(EdgarRateLimitError),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def _get(self, url: str, **kwargs: Any) -> httpx.Response:
        """Issue a rate-limited GET, retrying on 429."""
        client = self._assert_open()
        await self._limiter.acquire()
        resp = await client.get(url, **kwargs)
        if resp.status_code == 429:
            logger.warning("edgar_rate_limited", url=url)
            raise EdgarRateLimitError("EDGAR returned 429")
        if resp.status_code == 404:
            raise EdgarNotFoundError(f"Not found: {url}")
        resp.raise_for_status()
        return resp

    def _cache_key(self, url: str) -> Path | None:
        if not self._cache_dir:
            return None
        digest = hashlib.sha256(url.encode()).hexdigest()[:16]
        return self._cache_dir / f"{digest}.html"

    async def _get_cached(self, url: str) -> str:
        """Fetch *url*, returning cached content when available."""
        cache_path = self._cache_key(url)
        if cache_path and cache_path.exists():
            logger.debug("edgar_cache_hit", url=url)
            return cache_path.read_text(encoding="utf-8")

        resp = await self._get(url)
        text = resp.text
        if cache_path:
            cache_path.write_text(text, encoding="utf-8")
        return text

    # -- Public API -----------------------------------------------------------

    async def resolve_cik(self, ticker: str) -> str:
        """Resolve a ticker symbol to its CIK (zero-padded to 10 digits).

        Args:
            ticker: Upper-case ticker symbol.

        Returns:
            Zero-padded CIK string.

        Raises:
            EdgarNotFoundError: If the ticker cannot be resolved.
        """
        ticker = ticker.upper()
        if ticker in self._ticker_to_cik:
            return self._ticker_to_cik[ticker]

        resp = await self._get(_COMPANY_TICKERS_URL)
        data: dict[str, Any] = resp.json()
        for entry in data.values():
            t = str(entry.get("ticker", "")).upper()
            cik = str(entry.get("cik_str", "")).zfill(10)
            self._ticker_to_cik[t] = cik

        if ticker not in self._ticker_to_cik:
            raise EdgarNotFoundError(f"Unknown ticker: {ticker}")
        return self._ticker_to_cik[ticker]

    async def get_filings(
        self,
        ticker: str,
        filing_types: Sequence[str | FilingType] | None = None,
        lookback_years: int = 3,
    ) -> list[Filing]:
        """Retrieve filing metadata for *ticker* from EDGAR.

        Args:
            ticker: Company ticker symbol.
            filing_types: Restrict to these filing types.
                Defaults to ``["10-K", "10-Q"]``.
            lookback_years: How many years of filings to fetch.

        Returns:
            List of :class:`Filing` instances (without ``raw_html``).
        """
        cik = await self.resolve_cik(ticker)
        url = f"{_EDGAR_SUBMISSIONS}/CIK{cik}.json"
        resp = await self._get(url)
        payload: dict[str, Any] = resp.json()

        company_name = str(payload.get("name", ticker))
        recent = payload.get("filings", {}).get("recent", {})

        forms: list[str] = recent.get("form", [])
        accessions: list[str] = recent.get("accessionNumber", [])
        dates_filed: list[str] = recent.get("filingDate", [])
        periods: list[str] = recent.get("reportDate", [])
        primary_docs: list[str] = recent.get("primaryDocument", [])

        type_filter = set()
        if filing_types:
            for ft in filing_types:
                type_filter.add(ft.value if isinstance(ft, FilingType) else ft)
        else:
            type_filter = {"10-K", "10-Q"}

        cutoff = date.today() - timedelta(days=lookback_years * 365)
        filings: list[Filing] = []

        for i, form in enumerate(forms):
            if form not in type_filter:
                continue
            filed = date.fromisoformat(dates_filed[i])
            if filed < cutoff:
                continue

            accession_raw = accessions[i]
            accession_no_dash = accession_raw.replace("-", "")
            doc = primary_docs[i]
            filing_url = (
                f"{_EDGAR_ARCHIVES}/{cik.lstrip('0')}/{accession_no_dash}/{doc}"
            )

            period = date.fromisoformat(periods[i]) if periods[i] else filed

            try:
                ftype = FilingType(form)
            except ValueError:
                continue

            filings.append(
                Filing(
                    accession_number=accession_raw,
                    cik=cik,
                    ticker=ticker.upper(),
                    company_name=company_name,
                    filing_type=ftype,
                    filed_date=filed,
                    period_of_report=period,
                    url=filing_url,
                )
            )

        logger.info(
            "edgar_filings_found",
            ticker=ticker,
            count=len(filings),
            types=sorted(type_filter),
        )
        return sorted(filings, key=lambda f: f.filed_date)

    async def fetch_filing_html(self, filing: Filing) -> Filing:
        """Download the raw HTML for a filing and return an updated copy.

        Args:
            filing: A :class:`Filing` whose ``url`` points to EDGAR.

        Returns:
            New ``Filing`` with ``raw_html`` populated.
        """
        html = await self._get_cached(filing.url)
        return filing.model_copy(update={"raw_html": html})

    async def search_full_text(
        self,
        query: str,
        *,
        filing_types: Sequence[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Use EDGAR full-text search (EFTS) to find filings by keyword.

        .. deprecated::
            This method is **experimental**.  The EFTS API contract is
            not publicly documented by the SEC and may change without
            notice.  Use at your own risk and verify results manually.

        Args:
            query: Search query string.
            filing_types: Optional list of form types to restrict results.
            start_date: Earliest filing date.
            end_date: Latest filing date.
            limit: Maximum results to return.

        Returns:
            List of raw result dicts from the EFTS API.
        """
        import warnings

        warnings.warn(
            "search_full_text() is experimental. The EFTS API contract "
            "is not publicly documented and may change without notice.",
            stacklevel=2,
        )

        params: dict[str, Any] = {
            "q": query,
            "dateRange": "custom",
            "startdt": (
                start_date.isoformat()
                if start_date
                else (date.today() - timedelta(days=365)).isoformat()
            ),
            "enddt": (end_date.isoformat() if end_date else date.today().isoformat()),
        }
        if filing_types:
            params["forms"] = ",".join(filing_types)

        url = _EDGAR_FULL_TEXT_SEARCH
        resp = await self._get(url, params=params)
        data: dict[str, Any] = resp.json()
        hits: list[dict[str, Any]] = data.get("hits", {}).get("hits", [])
        return hits[:limit]
