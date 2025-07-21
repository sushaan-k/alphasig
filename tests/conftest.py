"""Shared test fixtures for sigint."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from sigint.models import (
    Filing,
    FilingSection,
    FilingType,
    RelationType,
    Signal,
    SignalDirection,
    SignalType,
    SupplyChainEdge,
)


@pytest.fixture
def sample_filing() -> Filing:
    """A minimal Filing instance for testing."""
    return Filing(
        accession_number="0000320193-24-000123",
        cik="0000320193",
        ticker="AAPL",
        company_name="Apple Inc.",
        filing_type=FilingType.TEN_K,
        filed_date=date(2024, 11, 1),
        period_of_report=date(2024, 9, 28),
        url="https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm",
        raw_html="<html><body>Test content</body></html>",
    )


@pytest.fixture
def sample_filing_with_sections() -> Filing:
    """A filing with realistic HTML containing identifiable sections."""
    html = """
    <html><body>
    <b>Item 1. Business</b>
    <p>Apple Inc. designs, manufactures, and markets smartphones,
    personal computers, tablets, wearables, and accessories worldwide.
    The Company relies on TSMC for semiconductor manufacturing and
    Foxconn for device assembly. Apple's supply chain spans multiple
    countries including China, Taiwan, and India. The company has
    significant partnerships with Qualcomm for modem chips and
    Broadcom for wireless components.</p>

    <b>Item 1A. Risk Factors</b>
    <p>The Company is subject to various risks including supply chain
    concentration risk. A significant portion of our components are
    sourced from a limited number of suppliers, including Taiwan
    Semiconductor Manufacturing Company (TSMC). Any disruption to
    TSMC's operations could materially affect our ability to deliver
    products. We face increased regulatory scrutiny in the European
    Union regarding our App Store policies. Geopolitical tensions
    between China and Taiwan pose risks to our supply chain. We may
    face antitrust enforcement actions in multiple jurisdictions.</p>

    <b>Item 7. Management's Discussion and Analysis</b>
    <p>Revenue for fiscal year 2024 increased 5% year over year,
    driven by strong iPhone sales and services growth. We are
    confident in our AI strategy and continue to expand our
    investment in machine learning capabilities. Gross margins
    improved to 46.2% from 44.1% in the prior year. We expect
    continued growth in our services segment. Our capital expenditure
    plans include significant investment in AI infrastructure.</p>

    <b>Item 8. Financial Statements</b>
    <p>Consolidated statements of operations for the fiscal year...</p>
    </body></html>
    """
    return Filing(
        accession_number="0000320193-24-000123",
        cik="0000320193",
        ticker="AAPL",
        company_name="Apple Inc.",
        filing_type=FilingType.TEN_K,
        filed_date=date(2024, 11, 1),
        period_of_report=date(2024, 9, 28),
        url="https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm",
        raw_html=html,
    )


@pytest.fixture
def sample_sections() -> list[FilingSection]:
    """Pre-parsed filing sections for engine testing."""
    return [
        FilingSection(
            filing_accession="0000320193-24-000123",
            ticker="AAPL",
            section_name="Business",
            section_key="business",
            text=(
                "Apple Inc. designs, manufactures, and markets smartphones. "
                "The Company relies on TSMC for semiconductor manufacturing "
                "and Foxconn for device assembly."
            ),
            filing_type=FilingType.TEN_K,
            filed_date=date(2024, 11, 1),
        ),
        FilingSection(
            filing_accession="0000320193-24-000123",
            ticker="AAPL",
            section_name="Risk Factors",
            section_key="risk_factors",
            text=(
                "We face supply chain concentration risk. A significant "
                "portion of components are sourced from TSMC. Disruptions "
                "could materially affect product delivery. We may face "
                "regulatory scrutiny in the EU."
            ),
            filing_type=FilingType.TEN_K,
            filed_date=date(2024, 11, 1),
        ),
        FilingSection(
            filing_accession="0000320193-24-000123",
            ticker="AAPL",
            section_name="Management Discussion and Analysis",
            section_key="md_and_a",
            text=(
                "Revenue increased 5% year over year. We are confident "
                "in our AI strategy. Gross margins improved to 46.2%."
            ),
            filing_type=FilingType.TEN_K,
            filed_date=date(2024, 11, 1),
        ),
    ]


@pytest.fixture
def previous_sections() -> list[FilingSection]:
    """Prior-period filing sections for diff engines."""
    return [
        FilingSection(
            filing_accession="0000320193-23-000100",
            ticker="AAPL",
            section_name="Risk Factors",
            section_key="risk_factors",
            text=(
                "We face supply chain risks. Components are sourced from "
                "multiple suppliers including TSMC. We could face regulatory "
                "inquiries in Europe."
            ),
            filing_type=FilingType.TEN_K,
            filed_date=date(2023, 11, 3),
        ),
        FilingSection(
            filing_accession="0000320193-23-000100",
            ticker="AAPL",
            section_name="Management Discussion and Analysis",
            section_key="md_and_a",
            text=(
                "Revenue was flat year over year. We are cautiously "
                "optimistic about our AI investments. Gross margins "
                "held steady at 44.1%."
            ),
            filing_type=FilingType.TEN_K,
            filed_date=date(2023, 11, 3),
        ),
    ]


@pytest.fixture
def sample_signals() -> list[Signal]:
    """A set of signals covering different types and directions."""
    return [
        Signal(
            timestamp=datetime(2024, 11, 1, tzinfo=UTC),
            ticker="AAPL",
            signal_type=SignalType.SUPPLY_CHAIN,
            direction=SignalDirection.NEUTRAL,
            strength=0.95,
            confidence=0.92,
            context="AAPL depends_on TSMC (semiconductor manufacturing)",
            source_filing="https://sec.gov/example",
            related_tickers=["TSMC"],
            metadata={
                "target": "TSMC",
                "relation": "depends_on",
                "edge_context": "semiconductor manufacturing",
                "filing_type": "10-K",
            },
        ),
        Signal(
            timestamp=datetime(2024, 11, 1, tzinfo=UTC),
            ticker="AAPL",
            signal_type=SignalType.RISK_CHANGE,
            direction=SignalDirection.BEARISH,
            strength=0.75,
            confidence=0.85,
            context="ESCALATED: Supply chain concentration risk",
            source_filing="https://sec.gov/example",
            related_tickers=["TSMC"],
            metadata={
                "change_type": "ESCALATED",
                "severity": "HIGH",
            },
        ),
        Signal(
            timestamp=datetime(2024, 11, 1, tzinfo=UTC),
            ticker="MSFT",
            signal_type=SignalType.M_AND_A,
            direction=SignalDirection.BULLISH,
            strength=0.80,
            confidence=0.70,
            context="M&A indicators detected: strategic alternatives",
            source_filing="https://sec.gov/example2",
            related_tickers=[],
            metadata={"indicator_count": 2},
        ),
        Signal(
            timestamp=datetime(2024, 11, 1, tzinfo=UTC),
            ticker="META",
            signal_type=SignalType.TONE_SHIFT,
            direction=SignalDirection.BEARISH,
            strength=0.60,
            confidence=0.78,
            context="Tone on 'AI spending' shifted from confident_expanding to hedging_cautious",
            source_filing="https://sec.gov/example3",
            related_tickers=[],
            metadata={
                "topic": "AI spending",
                "current_tone": "hedging_cautious",
                "previous_tone": "confident_expanding",
            },
        ),
    ]


@pytest.fixture
def sample_edges() -> list[SupplyChainEdge]:
    """Supply-chain edges for graph testing."""
    return [
        SupplyChainEdge(
            source="AAPL",
            target="TSMC",
            relation=RelationType.DEPENDS_ON,
            context="semiconductor manufacturing",
            confidence=0.95,
            filing_type=FilingType.TEN_K,
            filed_date=date(2024, 11, 1),
        ),
        SupplyChainEdge(
            source="AAPL",
            target="Foxconn",
            relation=RelationType.DEPENDS_ON,
            context="device assembly",
            confidence=0.92,
            filing_type=FilingType.TEN_K,
            filed_date=date(2024, 11, 1),
        ),
        SupplyChainEdge(
            source="NVDA",
            target="TSMC",
            relation=RelationType.DEPENDS_ON,
            context="GPU fabrication",
            confidence=0.97,
            filing_type=FilingType.TEN_K,
            filed_date=date(2024, 11, 1),
        ),
        SupplyChainEdge(
            source="AMD",
            target="TSMC",
            relation=RelationType.DEPENDS_ON,
            context="chip manufacturing",
            confidence=0.94,
            filing_type=FilingType.TEN_K,
            filed_date=date(2024, 11, 1),
        ),
    ]
