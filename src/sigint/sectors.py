"""Sector classification for publicly traded tickers.

Provides a :class:`Sector` enum and a :func:`classify_sector` helper that
maps ticker symbols to their GICS-inspired sector using a built-in lookup
table of ~120 major US-listed companies.
"""

from __future__ import annotations

import enum


class Sector(enum.StrEnum):
    """GICS-inspired sector classification."""

    TECHNOLOGY = "technology"
    HEALTHCARE = "healthcare"
    FINANCIALS = "financials"
    CONSUMER_DISCRETIONARY = "consumer_discretionary"
    CONSUMER_STAPLES = "consumer_staples"
    INDUSTRIALS = "industrials"
    ENERGY = "energy"
    UTILITIES = "utilities"
    MATERIALS = "materials"
    REAL_ESTATE = "real_estate"
    COMMUNICATION_SERVICES = "communication_services"
    UNKNOWN = "unknown"


# fmt: off
_TICKER_SECTOR_MAP: dict[str, Sector] = {
    # Technology
    "AAPL": Sector.TECHNOLOGY,
    "MSFT": Sector.TECHNOLOGY,
    "NVDA": Sector.TECHNOLOGY,
    "AVGO": Sector.TECHNOLOGY,
    "ORCL": Sector.TECHNOLOGY,
    "CRM": Sector.TECHNOLOGY,
    "AMD": Sector.TECHNOLOGY,
    "ADBE": Sector.TECHNOLOGY,
    "ACN": Sector.TECHNOLOGY,
    "CSCO": Sector.TECHNOLOGY,
    "INTC": Sector.TECHNOLOGY,
    "IBM": Sector.TECHNOLOGY,
    "TXN": Sector.TECHNOLOGY,
    "QCOM": Sector.TECHNOLOGY,
    "INTU": Sector.TECHNOLOGY,
    "AMAT": Sector.TECHNOLOGY,
    "NOW": Sector.TECHNOLOGY,
    "MU": Sector.TECHNOLOGY,
    "LRCX": Sector.TECHNOLOGY,
    "ADI": Sector.TECHNOLOGY,
    "KLAC": Sector.TECHNOLOGY,
    "SNPS": Sector.TECHNOLOGY,
    "CDNS": Sector.TECHNOLOGY,
    "CRWD": Sector.TECHNOLOGY,
    "PANW": Sector.TECHNOLOGY,
    "MRVL": Sector.TECHNOLOGY,
    "MSI": Sector.TECHNOLOGY,
    "FTNT": Sector.TECHNOLOGY,
    "HPQ": Sector.TECHNOLOGY,
    "DELL": Sector.TECHNOLOGY,
    # Communication Services
    "GOOGL": Sector.COMMUNICATION_SERVICES,
    "GOOG": Sector.COMMUNICATION_SERVICES,
    "META": Sector.COMMUNICATION_SERVICES,
    "NFLX": Sector.COMMUNICATION_SERVICES,
    "DIS": Sector.COMMUNICATION_SERVICES,
    "CMCSA": Sector.COMMUNICATION_SERVICES,
    "TMUS": Sector.COMMUNICATION_SERVICES,
    "VZ": Sector.COMMUNICATION_SERVICES,
    "T": Sector.COMMUNICATION_SERVICES,
    "CHTR": Sector.COMMUNICATION_SERVICES,
    "EA": Sector.COMMUNICATION_SERVICES,
    "WBD": Sector.COMMUNICATION_SERVICES,
    # Consumer Discretionary
    "AMZN": Sector.CONSUMER_DISCRETIONARY,
    "TSLA": Sector.CONSUMER_DISCRETIONARY,
    "HD": Sector.CONSUMER_DISCRETIONARY,
    "MCD": Sector.CONSUMER_DISCRETIONARY,
    "NKE": Sector.CONSUMER_DISCRETIONARY,
    "LOW": Sector.CONSUMER_DISCRETIONARY,
    "SBUX": Sector.CONSUMER_DISCRETIONARY,
    "TJX": Sector.CONSUMER_DISCRETIONARY,
    "BKNG": Sector.CONSUMER_DISCRETIONARY,
    "CMG": Sector.CONSUMER_DISCRETIONARY,
    "ABNB": Sector.CONSUMER_DISCRETIONARY,
    "ORLY": Sector.CONSUMER_DISCRETIONARY,
    "GM": Sector.CONSUMER_DISCRETIONARY,
    "F": Sector.CONSUMER_DISCRETIONARY,
    "ROST": Sector.CONSUMER_DISCRETIONARY,
    # Consumer Staples
    "WMT": Sector.CONSUMER_STAPLES,
    "PG": Sector.CONSUMER_STAPLES,
    "COST": Sector.CONSUMER_STAPLES,
    "KO": Sector.CONSUMER_STAPLES,
    "PEP": Sector.CONSUMER_STAPLES,
    "PM": Sector.CONSUMER_STAPLES,
    "MDLZ": Sector.CONSUMER_STAPLES,
    "CL": Sector.CONSUMER_STAPLES,
    "MO": Sector.CONSUMER_STAPLES,
    "TGT": Sector.CONSUMER_STAPLES,
    # Healthcare
    "UNH": Sector.HEALTHCARE,
    "LLY": Sector.HEALTHCARE,
    "JNJ": Sector.HEALTHCARE,
    "ABBV": Sector.HEALTHCARE,
    "MRK": Sector.HEALTHCARE,
    "TMO": Sector.HEALTHCARE,
    "ABT": Sector.HEALTHCARE,
    "DHR": Sector.HEALTHCARE,
    "PFE": Sector.HEALTHCARE,
    "AMGN": Sector.HEALTHCARE,
    "BMY": Sector.HEALTHCARE,
    "GILD": Sector.HEALTHCARE,
    "ISRG": Sector.HEALTHCARE,
    "VRTX": Sector.HEALTHCARE,
    "MDT": Sector.HEALTHCARE,
    "CI": Sector.HEALTHCARE,
    "ELV": Sector.HEALTHCARE,
    # Financials
    "BRK.B": Sector.FINANCIALS,
    "JPM": Sector.FINANCIALS,
    "V": Sector.FINANCIALS,
    "MA": Sector.FINANCIALS,
    "BAC": Sector.FINANCIALS,
    "WFC": Sector.FINANCIALS,
    "GS": Sector.FINANCIALS,
    "MS": Sector.FINANCIALS,
    "AXP": Sector.FINANCIALS,
    "BLK": Sector.FINANCIALS,
    "C": Sector.FINANCIALS,
    "SCHW": Sector.FINANCIALS,
    "SPGI": Sector.FINANCIALS,
    "CB": Sector.FINANCIALS,
    "MMC": Sector.FINANCIALS,
    # Industrials
    "CAT": Sector.INDUSTRIALS,
    "GE": Sector.INDUSTRIALS,
    "RTX": Sector.INDUSTRIALS,
    "HON": Sector.INDUSTRIALS,
    "UNP": Sector.INDUSTRIALS,
    "BA": Sector.INDUSTRIALS,
    "DE": Sector.INDUSTRIALS,
    "LMT": Sector.INDUSTRIALS,
    "UPS": Sector.INDUSTRIALS,
    "ADP": Sector.INDUSTRIALS,
    # Energy
    "XOM": Sector.ENERGY,
    "CVX": Sector.ENERGY,
    "COP": Sector.ENERGY,
    "SLB": Sector.ENERGY,
    "EOG": Sector.ENERGY,
    "MPC": Sector.ENERGY,
    "PSX": Sector.ENERGY,
    "VLO": Sector.ENERGY,
    "OXY": Sector.ENERGY,
    "WMB": Sector.ENERGY,
    # Utilities
    "NEE": Sector.UTILITIES,
    "SO": Sector.UTILITIES,
    "DUK": Sector.UTILITIES,
    "D": Sector.UTILITIES,
    "AEP": Sector.UTILITIES,
    "SRE": Sector.UTILITIES,
    # Materials
    "LIN": Sector.MATERIALS,
    "APD": Sector.MATERIALS,
    "SHW": Sector.MATERIALS,
    "ECL": Sector.MATERIALS,
    "NEM": Sector.MATERIALS,
    "FCX": Sector.MATERIALS,
    # Real Estate
    "PLD": Sector.REAL_ESTATE,
    "AMT": Sector.REAL_ESTATE,
    "EQIX": Sector.REAL_ESTATE,
    "SPG": Sector.REAL_ESTATE,
    "O": Sector.REAL_ESTATE,
    "CCI": Sector.REAL_ESTATE,
}
# fmt: on


def classify_sector(ticker: str) -> Sector:
    """Return the sector for a given ticker symbol.

    Uses a built-in lookup table of major US-listed companies.
    Returns :attr:`Sector.UNKNOWN` for unrecognised tickers.

    Args:
        ticker: Upper-case ticker symbol.

    Returns:
        The :class:`Sector` enum member.
    """
    return _TICKER_SECTOR_MAP.get(ticker.upper(), Sector.UNKNOWN)
