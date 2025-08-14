"""sigint -- Causal signal extraction from SEC filings.

LLM-powered pipeline that ingests SEC filings via EDGAR, performs deep
causal and structural extraction, and outputs structured, backtestable
signals.

Quick start::

    from sigint import Pipeline, SignalCollection

    pipeline = Pipeline(model="claude-sonnet-4-6")
    signals = await pipeline.extract(
        tickers=["AAPL", "MSFT"],
        filing_types=["10-K", "10-Q"],
        lookback_years=3,
    )

    # Filter and export
    bearish = signals.by_direction("bearish").above_strength(0.7)
    bearish.to_parquet("bearish_signals.parquet")
"""

from sigint.edgar import EdgarClient
from sigint.graph import SupplyChainGraph
from sigint.models import (
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
from sigint.pipeline import Pipeline
from sigint.sectors import Sector, classify_sector
from sigint.signals import CorrelationMatrix, SignalCollection
from sigint.storage import SignalStore

__version__ = "0.1.0"

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
    "Severity",
    "Signal",
    "SignalCollection",
    "SignalDirection",
    "SignalStore",
    "SignalType",
    "SupplyChainEdge",
    "SupplyChainGraph",
    "ToneLabel",
    "TonePoint",
    "ToneTrajectory",
    "classify_sector",
]
