"""Custom exception hierarchy for sigint.

All sigint-specific exceptions inherit from ``SigintError`` so callers
can catch the entire family with a single handler when desired.
"""

from __future__ import annotations


class SigintError(Exception):
    """Base exception for all sigint errors."""


class EdgarError(SigintError):
    """Error communicating with the SEC EDGAR API."""


class EdgarRateLimitError(EdgarError):
    """Raised when EDGAR rate-limits our requests (HTTP 429)."""


class EdgarNotFoundError(EdgarError):
    """Requested filing or entity does not exist on EDGAR."""


class ParsingError(SigintError):
    """Failed to parse a filing into structured sections."""


class ExtractionError(SigintError):
    """An extraction engine could not process its input."""


class LLMError(SigintError):
    """Error calling the LLM provider."""


class LLMRateLimitError(LLMError):
    """LLM provider returned a rate-limit error."""


class LLMContextLengthError(LLMError):
    """Input exceeded the model's context window."""


class StorageError(SigintError):
    """Error reading from or writing to DuckDB / Parquet."""


class PipelineError(SigintError):
    """Orchestration-level failure in the pipeline."""


class ConfigurationError(SigintError):
    """Invalid or missing configuration."""
