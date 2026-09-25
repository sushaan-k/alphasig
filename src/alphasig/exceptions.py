"""Custom exception hierarchy for alphasig.

All alphasig-specific exceptions inherit from ``AlphasigError`` so callers
can catch the entire family with a single handler when desired.
"""

from __future__ import annotations


class AlphasigError(Exception):
    """Base exception for all alphasig errors."""


class EdgarError(AlphasigError):
    """Error communicating with the SEC EDGAR API."""


class EdgarRateLimitError(EdgarError):
    """Raised when EDGAR rate-limits our requests (HTTP 429)."""

    retry_after: float | None = None


class EdgarTransientError(EdgarError):
    """Raised on transient network errors (timeouts, 5xx) that should be retried."""

    retry_after: float | None = None


class EdgarNotFoundError(EdgarError):
    """Requested filing or entity does not exist on EDGAR."""


class ParsingError(AlphasigError):
    """Failed to parse a filing into structured sections."""


class ExtractionError(AlphasigError):
    """An extraction engine could not process its input."""


class LLMError(AlphasigError):
    """Error calling the LLM provider."""


class LLMRateLimitError(LLMError):
    """LLM provider returned a rate-limit error."""


class LLMContextLengthError(LLMError):
    """Input exceeded the model's context window."""


class StorageError(AlphasigError):
    """Error reading from or writing to DuckDB / Parquet."""


class PipelineError(AlphasigError):
    """Orchestration-level failure in the pipeline."""


class ConfigurationError(AlphasigError):
    """Invalid or missing configuration."""
