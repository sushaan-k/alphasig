"""Async EDGAR API client with rate limiting and caching.

SEC EDGAR's fair-access policy requires a declared User-Agent with a
contact email and at most 10 requests per second.  One client shares a
single connection pool and rate limiter across all concurrent tasks, backs
off on 429 / 5xx (honouring ``Retry-After``), and times out stalled
requests.

Only the form types in :class:`~alphasig.models.FilingType` are returned;
amendments (``10-K/A`` etc.) are skipped because the original filing is
what the market saw first, and many amendments only restate Part III.

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
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import structlog
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from alphasig.exceptions import (
    EdgarError,
    EdgarNotFoundError,
    EdgarRateLimitError,
    EdgarTransientError,
)
from alphasig.models import Filing, FilingType

_EASTERN = ZoneInfo("America/New_York")

logger = structlog.get_logger()

_EDGAR_FULL_TEXT_SEARCH = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions"
_EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

_MAX_RPS = 10  # SEC rate limit


class _RateLimiter:
    """Spaces requests at least ``1 / max_per_second`` apart.

    Unlike a token bucket this never bursts, so no one-second window can
    exceed the SEC's limit.  Waiters reserve their slot under the lock and
    sleep outside it, so concurrent tasks sharing one client queue fairly.
    """

    def __init__(self, max_per_second: int = _MAX_RPS) -> None:
        self._interval = 1.0 / max_per_second
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self._interval
        if slot > now:
            await asyncio.sleep(slot - now)


def _retry_after_seconds(resp: httpx.Response) -> float | None:
    """Parse a numeric ``Retry-After`` header (HTTP-date form is ignored)."""
    try:
        return max(0.0, float(resp.headers["Retry-After"]))
    except (KeyError, ValueError):
        return None


_backoff = wait_exponential(multiplier=1, min=2, max=30)


def _retry_wait(state: RetryCallState) -> float:
    """Honour the server's ``Retry-After`` when given, else back off exponentially."""
    exc = state.outcome.exception() if state.outcome else None
    retry_after = getattr(exc, "retry_after", None)
    if retry_after is not None:
        return float(min(retry_after, 60.0))
    return float(_backoff(state))


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
        self._cik_lock = asyncio.Lock()

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
        retry=retry_if_exception_type((EdgarRateLimitError, EdgarTransientError)),
        wait=_retry_wait,
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def _get(self, url: str, **kwargs: Any) -> httpx.Response:
        """Issue a rate-limited GET, retrying on 429, network errors, and 5xx."""
        client = self._assert_open()
        await self._limiter.acquire()
        try:
            resp = await client.get(url, **kwargs)
        except httpx.TransportError as exc:
            logger.warning("edgar_network_error", url=url, error=str(exc))
            raise EdgarTransientError(f"Network error fetching {url}: {exc}") from exc
        if resp.status_code == 429:
            logger.warning("edgar_rate_limited", url=url)
            err = EdgarRateLimitError("EDGAR returned 429")
            err.retry_after = _retry_after_seconds(resp)
            raise err
        if resp.status_code == 404:
            raise EdgarNotFoundError(f"Not found: {url}")
        if resp.status_code >= 500:
            logger.warning("edgar_server_error", url=url, status=resp.status_code)
            transient = EdgarTransientError(
                f"EDGAR returned {resp.status_code} for {url}"
            )
            transient.retry_after = _retry_after_seconds(resp)
            raise transient
        if resp.status_code == 403:
            raise EdgarError(
                f"EDGAR refused {url} (403). The SEC blocks requests without a "
                "declared User-Agent ('Name email@domain') and clients that "
                "exceed 10 requests/second."
            )
        if resp.is_error:
            raise EdgarError(f"EDGAR returned {resp.status_code} for {url}")
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
            # Write-then-rename so an interrupted run never leaves a truncated
            # file that later reads would treat as a cache hit.
            tmp_path = cache_path.with_suffix(".tmp")
            tmp_path.write_text(text, encoding="utf-8")
            tmp_path.replace(cache_path)
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
        # The lock stops concurrent tickers from each downloading the
        # (large) ticker map before the first download has populated it.
        async with self._cik_lock:
            if not self._ticker_to_cik:
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
        resp = await self._get(f"{_EDGAR_SUBMISSIONS}/CIK{cik}.json")
        payload: dict[str, Any] = resp.json()

        company_name = str(payload.get("name", ticker))
        filings_meta = payload.get("filings", {})

        type_filter = (
            {ft.value if isinstance(ft, FilingType) else ft for ft in filing_types}
            if filing_types
            else {"10-K", "10-Q"}
        )
        unsupported = type_filter - {ft.value for ft in FilingType}
        if unsupported:
            logger.warning("edgar_unsupported_form_types", types=sorted(unsupported))

        cutoff = date.today() - timedelta(days=lookback_years * 365)

        # ``recent`` holds only the latest ~1000 filings; prolific filers
        # (e.g. heavy Form 4 activity) push older 10-K/10-Qs into
        # additional pages listed under ``files``.
        pages: list[dict[str, Any]] = [filings_meta.get("recent", {})]
        for extra in filings_meta.get("files", []):
            filing_to = extra.get("filingTo")
            if filing_to and date.fromisoformat(filing_to) >= cutoff:
                older = await self._get(f"{_EDGAR_SUBMISSIONS}/{extra['name']}")
                pages.append(older.json())

        filings: list[Filing] = []
        for page in pages:
            filings.extend(
                self._filings_from_page(
                    page, cik, ticker.upper(), company_name, type_filter, cutoff
                )
            )

        logger.info(
            "edgar_filings_found",
            ticker=ticker,
            count=len(filings),
            types=sorted(type_filter),
        )
        return sorted(filings, key=lambda f: f.filed_date)

    @staticmethod
    def _filings_from_page(
        page: dict[str, Any],
        cik: str,
        ticker: str,
        company_name: str,
        type_filter: set[str],
        cutoff: date,
    ) -> list[Filing]:
        """Build :class:`Filing` objects from one columnar submissions page."""
        forms: list[str] = page.get("form", [])
        accessions: list[str] = page.get("accessionNumber", [])
        dates_filed: list[str] = page.get("filingDate", [])
        periods: list[str] = page.get("reportDate", [])
        primary_docs: list[str] = page.get("primaryDocument", [])
        accepted: list[str] = page.get("acceptanceDateTime", [])

        filings: list[Filing] = []
        for i, form in enumerate(forms):
            if form not in type_filter:
                continue
            try:
                ftype = FilingType(form)
            except ValueError:
                continue
            filed = date.fromisoformat(dates_filed[i])
            if filed < cutoff or not primary_docs[i]:
                continue

            accession = accessions[i]
            filing_url = (
                f"{_EDGAR_ARCHIVES}/{int(cik)}/{accession.replace('-', '')}/"
                f"{primary_docs[i]}"
            )
            period = date.fromisoformat(periods[i]) if periods[i] else filed

            filings.append(
                Filing(
                    accession_number=accession,
                    cik=cik,
                    ticker=ticker,
                    company_name=company_name,
                    filing_type=ftype,
                    filed_date=filed,
                    period_of_report=period,
                    url=filing_url,
                    accepted_at=_parse_acceptance(
                        accepted[i] if i < len(accepted) else ""
                    ),
                )
            )
        return filings

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


def _parse_acceptance(raw: str) -> datetime | None:
    """Parse EDGAR's ``acceptanceDateTime``.

    The submissions API renders it with a ``Z`` suffix, but the value is
    Eastern time (it matches the "Accepted" time on EDGAR filing index
    pages).  Treating it as UTC would date filings up to five hours too
    early -- a look-ahead bias in any backtest keyed on it.
    """
    if not raw:
        return None
    try:
        naive = datetime.fromisoformat(raw.rstrip("Z")).replace(tzinfo=None)
    except ValueError:
        return None
    return naive.replace(tzinfo=_EASTERN)
