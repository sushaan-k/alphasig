"""alphasig -- Causal signal extraction from SEC filings.

LLM-powered pipeline that ingests SEC filings via EDGAR, performs deep
causal and structural extraction, and outputs structured, backtestable
signals.

Quick start::

    from alphasig import Pipeline, SignalCollection

    pipeline = Pipeline(user_agent="Jane Doe jane@example.com")
    signals = await pipeline.extract(
        tickers=["AAPL", "MSFT"],
        filing_types=["10-K", "10-Q"],
        lookback_years=3,
    )

    # Filter and export
    bearish = signals.by_direction("bearish").above_strength(0.7)
    bearish.to_parquet("bearish_signals.parquet")
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from alphasig.edgar import EdgarClient
from alphasig.graph import SupplyChainGraph
from alphasig.models import (
    Filing,
    FilingSection,
    FilingType,
    MandAIndicator,
    RiskChange,
    RiskChangeType,
    Severity,
    Signal,
    SignalDirection,
    SignalType,
    SupplyChainEdge,
    ToneLabel,
    TonePoint,
    ToneTrajectory,
)
from alphasig.pipeline import Pipeline
from alphasig.reporting import (
    SectorExposureReport,
    SectorScore,
    SignalRankingReport,
    TickerScore,
    rank_signals,
    summarize_sector_exposure,
)
from alphasig.sectors import Sector, classify_sector
from alphasig.signals import CorrelationMatrix, SignalCollection
from alphasig.storage import SignalStore

try:
    __version__ = _version("alphasig")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0+unknown"

__all__ = [
    "CorrelationMatrix",
    "EdgarClient",
    "Filing",
    "FilingSection",
    "FilingType",
    "MandAIndicator",
    "Pipeline",
    "RiskChange",
    "RiskChangeType",
    "Sector",
    "SectorExposureReport",
    "SectorScore",
    "Severity",
    "Signal",
    "SignalCollection",
    "SignalDirection",
    "SignalRankingReport",
    "SignalStore",
    "SignalType",
    "SupplyChainEdge",
    "SupplyChainGraph",
    "TickerScore",
    "ToneLabel",
    "TonePoint",
    "ToneTrajectory",
    "classify_sector",
    "rank_signals",
    "summarize_sector_exposure",
]
