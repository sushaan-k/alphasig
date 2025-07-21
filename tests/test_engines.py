"""Tests for extraction engines (mocking the LLM)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from sigint.engines.m_and_a import MandAEngine
from sigint.engines.risk_differ import RiskDifferEngine, compute_text_similarity
from sigint.engines.supply_chain import SupplyChainEngine
from sigint.engines.tone import ToneEngine, classify_tone_shift
from sigint.llm import LLMClient
from sigint.models import (
    FilingSection,
    SignalDirection,
    SignalType,
    ToneLabel,
)


@pytest.fixture
def mock_llm() -> LLMClient:
    """An LLMClient with a mocked extract_json method."""
    llm = MagicMock(spec=LLMClient)
    llm.extract_json = AsyncMock()
    llm.extract_model = AsyncMock()
    llm.extract_model_list = AsyncMock()
    return llm


# -- Supply Chain Engine -----------------------------------------------------


class TestSupplyChainEngine:
    """Tests for the SupplyChainEngine."""

    @pytest.mark.asyncio
    async def test_extracts_edges(
        self,
        mock_llm: LLMClient,
        sample_sections: list[FilingSection],
    ) -> None:
        mock_llm.extract_json.return_value = [  # type: ignore[attr-defined]
            {
                "source": "AAPL",
                "target": "TSMC",
                "relation": "depends_on",
                "context": "semiconductor manufacturing",
                "confidence": 0.95,
            },
            {
                "source": "AAPL",
                "target": "Foxconn",
                "relation": "depends_on",
                "context": "device assembly",
                "confidence": 0.92,
            },
        ]

        engine = SupplyChainEngine()
        signals = await engine.extract(sample_sections, mock_llm)

        assert len(signals) >= 2
        assert all(s.signal_type == SignalType.SUPPLY_CHAIN for s in signals)
        tickers = {s.metadata.get("target") for s in signals}
        assert "TSMC" in tickers

    @pytest.mark.asyncio
    async def test_deduplicates_edges(
        self,
        mock_llm: LLMClient,
        sample_sections: list[FilingSection],
    ) -> None:
        # Same edge returned from multiple sections
        mock_llm.extract_json.return_value = [  # type: ignore[attr-defined]
            {
                "source": "AAPL",
                "target": "TSMC",
                "relation": "depends_on",
                "context": "semiconductor manufacturing",
                "confidence": 0.95,
            },
        ]

        engine = SupplyChainEngine()
        signals = await engine.extract(sample_sections, mock_llm)

        # Should be deduplicated to 1 unique TSMC edge
        tsmc_signals = [s for s in signals if s.metadata.get("target") == "TSMC"]
        assert len(tsmc_signals) == 1

    @pytest.mark.asyncio
    async def test_no_sections_returns_empty(self, mock_llm: LLMClient) -> None:
        engine = SupplyChainEngine()
        signals = await engine.extract([], mock_llm)
        assert signals == []

    @pytest.mark.asyncio
    async def test_supply_chain_signal_metadata_preserves_filing_type(
        self,
        mock_llm: LLMClient,
        sample_sections: list[FilingSection],
    ) -> None:
        mock_llm.extract_json.return_value = [  # type: ignore[attr-defined]
            {
                "target": "TSMC",
                "relation": "depends_on",
                "context": "semiconductor manufacturing",
                "confidence": 0.95,
            },
        ]
        signals = await SupplyChainEngine().extract(sample_sections, mock_llm)
        assert signals[0].metadata["filing_type"] == "10-K"

    @pytest.mark.asyncio
    async def test_no_relevant_sections_returns_empty(
        self, mock_llm: LLMClient
    ) -> None:
        """Sections not in {business, risk_factors, md_and_a} are skipped."""
        from datetime import date

        from sigint.models import FilingType

        sections = [
            FilingSection(
                filing_accession="0001234-24-000001",
                ticker="AAPL",
                section_name="Properties",
                section_key="properties",
                text="Some property description",
                filing_type=FilingType.TEN_K,
                filed_date=date(2024, 11, 1),
            ),
        ]
        engine = SupplyChainEngine()
        signals = await engine.extract(sections, mock_llm)
        assert signals == []

    @pytest.mark.asyncio
    async def test_malformed_item_is_skipped(
        self,
        mock_llm: LLMClient,
        sample_sections: list[FilingSection],
    ) -> None:
        """Invalid items from LLM are skipped without crashing."""
        mock_llm.extract_json.return_value = [  # type: ignore[attr-defined]
            {
                "source": "AAPL",
                "target": "TSMC",
                "relation": "invalid_relation_type",
                "context": "test",
                "confidence": 0.9,
            },
            {
                "source": "AAPL",
                "target": "Foxconn",
                "relation": "depends_on",
                "context": "assembly",
                "confidence": 0.8,
            },
        ]
        engine = SupplyChainEngine()
        signals = await engine.extract(sample_sections, mock_llm)
        # First item is invalid, second is valid
        assert len(signals) == 1
        assert signals[0].metadata["target"] == "Foxconn"

    @pytest.mark.asyncio
    async def test_non_list_response_wrapped(
        self,
        mock_llm: LLMClient,
        sample_sections: list[FilingSection],
    ) -> None:
        """LLM returns a single dict instead of a list; should be wrapped."""
        mock_llm.extract_json.return_value = {  # type: ignore[attr-defined]
            "source": "AAPL",
            "target": "TSMC",
            "relation": "depends_on",
            "context": "chips",
            "confidence": 0.9,
        }
        engine = SupplyChainEngine()
        signals = await engine.extract(sample_sections, mock_llm)
        assert len(signals) >= 1

    def test_engine_name(self) -> None:
        assert SupplyChainEngine().name == "supply_chain"


# -- Risk Differ Engine ------------------------------------------------------


class TestRiskDifferEngine:
    """Tests for the RiskDifferEngine."""

    @pytest.mark.asyncio
    async def test_detects_changes(
        self,
        mock_llm: LLMClient,
        sample_sections: list[FilingSection],
        previous_sections: list[FilingSection],
    ) -> None:
        mock_llm.extract_json.return_value = [  # type: ignore[attr-defined]
            {
                "change_type": "ESCALATED",
                "risk": "Supply chain concentration in TSMC",
                "language_shift": "'risks' -> 'concentration risk'",
                "severity_estimate": "HIGH",
                "related_tickers": ["TSMC"],
            },
            {
                "change_type": "NEW",
                "risk": "Geopolitical tensions affecting supply chain",
                "language_shift": "",
                "severity_estimate": "CRITICAL",
                "related_tickers": [],
            },
        ]

        engine = RiskDifferEngine()
        signals = await engine.extract(
            sample_sections,
            mock_llm,
            previous_sections=previous_sections,
        )

        assert len(signals) == 2
        assert all(s.signal_type == SignalType.RISK_CHANGE for s in signals)
        # Both are bearish (ESCALATED and NEW)
        assert all(s.direction == SignalDirection.BEARISH for s in signals)

    @pytest.mark.asyncio
    async def test_no_previous_returns_empty(
        self,
        mock_llm: LLMClient,
        sample_sections: list[FilingSection],
    ) -> None:
        engine = RiskDifferEngine()
        signals = await engine.extract(sample_sections, mock_llm)
        assert signals == []

    def test_compute_text_similarity(self) -> None:
        assert compute_text_similarity("hello world", "hello world") == 1.0
        assert compute_text_similarity("hello", "goodbye") < 0.5
        sim = compute_text_similarity(
            "The company faces risks from supply chain disruption",
            "The company faces risks from supply chain concentration",
        )
        assert 0.7 < sim < 1.0

    @pytest.mark.asyncio
    async def test_empty_sections_returns_empty(self, mock_llm: LLMClient) -> None:
        engine = RiskDifferEngine()
        signals = await engine.extract([], mock_llm)
        assert signals == []

    @pytest.mark.asyncio
    async def test_no_risk_factors_in_current(
        self, mock_llm: LLMClient, previous_sections: list[FilingSection]
    ) -> None:
        """When current filing lacks risk_factors section, return empty."""
        from datetime import date

        from sigint.models import FilingType

        sections_without_rf = [
            FilingSection(
                filing_accession="0001234-24-000001",
                ticker="AAPL",
                section_name="Business",
                section_key="business",
                text="Some business text",
                filing_type=FilingType.TEN_K,
                filed_date=date(2024, 11, 1),
            ),
        ]
        engine = RiskDifferEngine()
        signals = await engine.extract(
            sections_without_rf, mock_llm, previous_sections=previous_sections
        )
        assert signals == []

    @pytest.mark.asyncio
    async def test_no_risk_factors_in_previous(
        self, mock_llm: LLMClient, sample_sections: list[FilingSection]
    ) -> None:
        """When previous filing lacks risk_factors, return empty."""
        from datetime import date

        from sigint.models import FilingType

        prev_without_rf = [
            FilingSection(
                filing_accession="0001234-23-000001",
                ticker="AAPL",
                section_name="Business",
                section_key="business",
                text="Old business text",
                filing_type=FilingType.TEN_K,
                filed_date=date(2023, 11, 1),
            ),
        ]
        engine = RiskDifferEngine()
        signals = await engine.extract(
            sample_sections, mock_llm, previous_sections=prev_without_rf
        )
        assert signals == []

    @pytest.mark.asyncio
    async def test_nearly_identical_text_skips_llm(
        self,
        mock_llm: LLMClient,
    ) -> None:
        """When similarity > 0.98, skip the LLM call entirely."""
        from datetime import date

        from sigint.models import FilingType

        text = "We face supply chain risks. " * 100
        current = [
            FilingSection(
                filing_accession="ACC-001",
                ticker="AAPL",
                section_name="Risk Factors",
                section_key="risk_factors",
                text=text,
                filing_type=FilingType.TEN_K,
                filed_date=date(2024, 11, 1),
            ),
        ]
        previous = [
            FilingSection(
                filing_accession="ACC-000",
                ticker="AAPL",
                section_name="Risk Factors",
                section_key="risk_factors",
                text=text,
                filing_type=FilingType.TEN_K,
                filed_date=date(2023, 11, 1),
            ),
        ]
        engine = RiskDifferEngine()
        signals = await engine.extract(current, mock_llm, previous_sections=previous)
        assert signals == []
        mock_llm.extract_json.assert_not_called()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_malformed_change_item_is_skipped(
        self,
        mock_llm: LLMClient,
        sample_sections: list[FilingSection],
        previous_sections: list[FilingSection],
    ) -> None:
        """Items with invalid change_type are skipped gracefully."""
        mock_llm.extract_json.return_value = [  # type: ignore[attr-defined]
            {
                "change_type": "INVALID_TYPE",
                "risk": "Something",
                "severity_estimate": "HIGH",
            },
            {
                "change_type": "NEW",
                "risk": "Valid risk",
                "severity_estimate": "MEDIUM",
                "related_tickers": [],
            },
        ]
        engine = RiskDifferEngine()
        signals = await engine.extract(
            sample_sections, mock_llm, previous_sections=previous_sections
        )
        assert len(signals) == 1

    def test_compute_text_similarity_empty(self) -> None:
        assert compute_text_similarity("", "") == 1.0

    def test_compute_text_similarity_whitespace_normalised(self) -> None:
        sim = compute_text_similarity("hello   world", "hello world")
        assert sim == 1.0

    def test_engine_name(self) -> None:
        assert RiskDifferEngine().name == "risk_differ"


# -- M&A Engine --------------------------------------------------------------


class TestMandAEngine:
    """Tests for the MandAEngine."""

    @pytest.mark.asyncio
    async def test_detects_indicators(
        self,
        mock_llm: LLMClient,
        sample_sections: list[FilingSection],
    ) -> None:
        mock_llm.extract_json.return_value = [  # type: ignore[attr-defined]
            {
                "indicator": "strategic alternatives language",
                "category": "strategic_alternatives",
                "excerpt": "The company is exploring strategic alternatives",
                "confidence": 0.88,
            },
        ]

        engine = MandAEngine()
        signals = await engine.extract(sample_sections, mock_llm)

        assert len(signals) >= 1
        assert signals[0].signal_type == SignalType.M_AND_A

    @pytest.mark.asyncio
    async def test_no_indicators_returns_empty(
        self,
        mock_llm: LLMClient,
        sample_sections: list[FilingSection],
    ) -> None:
        mock_llm.extract_json.return_value = []  # type: ignore[attr-defined]

        engine = MandAEngine()
        signals = await engine.extract(sample_sections, mock_llm)
        assert signals == []

    @pytest.mark.asyncio
    async def test_empty_sections_returns_empty(self, mock_llm: LLMClient) -> None:
        engine = MandAEngine()
        signals = await engine.extract([], mock_llm)
        assert signals == []

    @pytest.mark.asyncio
    async def test_high_confidence_strategic_alternatives_is_bullish(
        self,
        mock_llm: LLMClient,
        sample_sections: list[FilingSection],
    ) -> None:
        mock_llm.extract_json.return_value = [  # type: ignore[attr-defined]
            {
                "indicator": "exploring strategic alternatives",
                "category": "strategic_alternatives",
                "excerpt": "The board is exploring strategic alternatives",
                "confidence": 0.95,
            },
        ]
        engine = MandAEngine()
        signals = await engine.extract(sample_sections, mock_llm)
        assert len(signals) == 1
        assert signals[0].direction == SignalDirection.BULLISH

    @pytest.mark.asyncio
    async def test_low_confidence_strategic_alternatives_is_neutral(
        self,
        mock_llm: LLMClient,
        sample_sections: list[FilingSection],
    ) -> None:
        mock_llm.extract_json.return_value = [  # type: ignore[attr-defined]
            {
                "indicator": "mentioned alternatives",
                "category": "strategic_alternatives",
                "excerpt": "Some vague mention",
                "confidence": 0.3,
            },
        ]
        engine = MandAEngine()
        signals = await engine.extract(sample_sections, mock_llm)
        assert len(signals) == 1
        assert signals[0].direction == SignalDirection.NEUTRAL

    @pytest.mark.asyncio
    async def test_malformed_indicator_is_skipped(
        self,
        mock_llm: LLMClient,
        sample_sections: list[FilingSection],
    ) -> None:
        """MandAIndicator with invalid confidence should be skipped."""
        mock_llm.extract_json.return_value = [  # type: ignore[attr-defined]
            {
                "indicator": "test",
                "category": "strategic_alternatives",
                "excerpt": "text",
                "confidence": "not_a_number",
            },
            {
                "indicator": "valid",
                "category": "cash_positioning",
                "excerpt": "valid text",
                "confidence": 0.6,
            },
        ]
        engine = MandAEngine()
        signals = await engine.extract(sample_sections, mock_llm)
        # First is invalid (bad confidence), second is valid
        assert len(signals) == 1

    @pytest.mark.asyncio
    async def test_non_list_response_wrapped(
        self,
        mock_llm: LLMClient,
        sample_sections: list[FilingSection],
    ) -> None:
        """Single dict response from LLM should be wrapped in a list."""
        mock_llm.extract_json.return_value = {  # type: ignore[attr-defined]
            "indicator": "board changes",
            "category": "board_change",
            "excerpt": "New director with PE background",
            "confidence": 0.7,
        }
        engine = MandAEngine()
        signals = await engine.extract(sample_sections, mock_llm)
        assert len(signals) == 1

    @pytest.mark.asyncio
    async def test_multiple_categories_aggregated(
        self,
        mock_llm: LLMClient,
        sample_sections: list[FilingSection],
    ) -> None:
        # Return items only on first call, empty for remaining sections
        mock_llm.extract_json.side_effect = [  # type: ignore[attr-defined]
            [
                {
                    "indicator": "strategic review",
                    "category": "strategic_alternatives",
                    "excerpt": "Board exploring options",
                    "confidence": 0.9,
                },
                {
                    "indicator": "Goldman Sachs engaged",
                    "category": "advisor_engagement",
                    "excerpt": "Engaged Goldman",
                    "confidence": 0.85,
                },
                {
                    "indicator": "increased cash reserves",
                    "category": "cash_positioning",
                    "excerpt": "Cash grew to $5B",
                    "confidence": 0.7,
                },
            ],
            [],
            [],
        ]
        engine = MandAEngine()
        signals = await engine.extract(sample_sections, mock_llm)
        assert len(signals) == 1  # One summary signal
        assert signals[0].metadata["indicator_count"] == 3
        categories = signals[0].metadata["categories"]
        assert len(categories) == 3

    def test_engine_name(self) -> None:
        assert MandAEngine().name == "m_and_a"


# -- Tone Engine -------------------------------------------------------------


class TestToneEngine:
    """Tests for the ToneEngine."""

    @pytest.mark.asyncio
    async def test_detects_tone_shift(
        self,
        mock_llm: LLMClient,
        sample_sections: list[FilingSection],
        previous_sections: list[FilingSection],
    ) -> None:
        # First call is for current MDA, second for previous
        mock_llm.extract_json.side_effect = [  # type: ignore[attr-defined]
            [
                {
                    "topic": "AI strategy",
                    "tone": "confident_expanding",
                    "confidence": 0.85,
                    "key_phrases": ["confident in AI strategy"],
                },
            ],
            [
                {
                    "topic": "AI strategy",
                    "tone": "hedging_cautious",
                    "confidence": 0.70,
                    "key_phrases": ["cautiously optimistic"],
                },
            ],
        ]

        engine = ToneEngine()
        signals = await engine.extract(
            sample_sections,
            mock_llm,
            previous_sections=previous_sections,
        )

        assert len(signals) >= 1
        assert signals[0].signal_type == SignalType.TONE_SHIFT
        # Went from hedging_cautious to confident_expanding = bullish
        assert signals[0].direction == SignalDirection.BULLISH

    @pytest.mark.asyncio
    async def test_no_previous_sections_returns_no_tone_shift_signals(
        self,
        mock_llm: LLMClient,
        sample_sections: list[FilingSection],
    ) -> None:
        mock_llm.extract_json.return_value = [  # type: ignore[attr-defined]
            {
                "topic": "revenue growth",
                "tone": "confident_expanding",
                "confidence": 0.82,
                "key_phrases": ["strong growth"],
            },
        ]

        engine = ToneEngine()
        signals = await engine.extract(sample_sections, mock_llm)

        assert signals == []

    def test_classify_tone_shift(self) -> None:
        direction, strength = classify_tone_shift(
            ToneLabel.HEDGING_CAUTIOUS,
            ToneLabel.CONFIDENT_EXPANDING,
        )
        assert direction == SignalDirection.BEARISH
        assert strength > 0

    def test_classify_tone_no_shift(self) -> None:
        direction, strength = classify_tone_shift(
            ToneLabel.NEUTRAL_FACTUAL,
            ToneLabel.NEUTRAL_FACTUAL,
        )
        assert direction == SignalDirection.NEUTRAL
        assert strength == 0.0

    @pytest.mark.asyncio
    async def test_no_mda_section_returns_empty(self, mock_llm: LLMClient) -> None:
        from datetime import date

        from sigint.models import FilingType

        sections = [
            FilingSection(
                filing_accession="ACC-001",
                ticker="AAPL",
                section_name="Risk Factors",
                section_key="risk_factors",
                text="Some risk text",
                filing_type=FilingType.TEN_K,
                filed_date=date(2024, 11, 1),
            ),
        ]
        engine = ToneEngine()
        signals = await engine.extract(sections, mock_llm)
        assert signals == []

    @pytest.mark.asyncio
    async def test_empty_sections_returns_empty(self, mock_llm: LLMClient) -> None:
        engine = ToneEngine()
        signals = await engine.extract([], mock_llm)
        assert signals == []

    @pytest.mark.asyncio
    async def test_no_tones_extracted_returns_empty(
        self,
        mock_llm: LLMClient,
        sample_sections: list[FilingSection],
    ) -> None:
        mock_llm.extract_json.return_value = []  # type: ignore[attr-defined]
        engine = ToneEngine()
        signals = await engine.extract(sample_sections, mock_llm)
        assert signals == []

    @pytest.mark.asyncio
    async def test_invalid_tone_label_skipped(
        self,
        mock_llm: LLMClient,
        sample_sections: list[FilingSection],
    ) -> None:
        """Items with invalid tone labels should be skipped."""
        mock_llm.extract_json.return_value = [  # type: ignore[attr-defined]
            {
                "topic": "revenue",
                "tone": "completely_invalid_tone",
                "confidence": 0.8,
                "key_phrases": ["test"],
            },
        ]
        engine = ToneEngine()
        signals = await engine.extract(sample_sections, mock_llm)
        # Baseline with invalid tone should produce nothing
        assert signals == []

    @pytest.mark.asyncio
    async def test_empty_topic_skipped(
        self,
        mock_llm: LLMClient,
        sample_sections: list[FilingSection],
    ) -> None:
        mock_llm.extract_json.return_value = [  # type: ignore[attr-defined]
            {
                "topic": "",
                "tone": "confident_expanding",
                "confidence": 0.8,
            },
        ]
        engine = ToneEngine()
        signals = await engine.extract(sample_sections, mock_llm)
        assert signals == []

    @pytest.mark.asyncio
    async def test_no_matching_previous_topic_skipped(
        self,
        mock_llm: LLMClient,
        sample_sections: list[FilingSection],
        previous_sections: list[FilingSection],
    ) -> None:
        """When topics don't match between current and previous, no shift."""
        mock_llm.extract_json.side_effect = [  # type: ignore[attr-defined]
            [
                {
                    "topic": "AI spending",
                    "tone": "pessimistic_warning",
                    "confidence": 0.9,
                },
            ],
            [
                {
                    "topic": "totally different topic",
                    "tone": "confident_expanding",
                    "confidence": 0.9,
                },
            ],
        ]
        engine = ToneEngine()
        signals = await engine.extract(
            sample_sections, mock_llm, previous_sections=previous_sections
        )
        assert signals == []

    @pytest.mark.asyncio
    async def test_same_tone_no_shift(
        self,
        mock_llm: LLMClient,
        sample_sections: list[FilingSection],
        previous_sections: list[FilingSection],
    ) -> None:
        """Same tone in both filings means no signal emitted."""
        mock_llm.extract_json.side_effect = [  # type: ignore[attr-defined]
            [
                {
                    "topic": "revenue growth",
                    "tone": "neutral_factual",
                    "confidence": 0.8,
                },
            ],
            [
                {
                    "topic": "revenue growth",
                    "tone": "neutral_factual",
                    "confidence": 0.8,
                },
            ],
        ]
        engine = ToneEngine()
        signals = await engine.extract(
            sample_sections, mock_llm, previous_sections=previous_sections
        )
        assert signals == []

    def test_classify_tone_shift_bullish(self) -> None:
        direction, strength = classify_tone_shift(
            ToneLabel.CONFIDENT_EXPANDING,
            ToneLabel.PESSIMISTIC_WARNING,
        )
        assert direction == SignalDirection.BULLISH
        assert strength == 1.0  # Max shift of 5 steps

    def test_classify_tone_shift_one_step(self) -> None:
        direction, strength = classify_tone_shift(
            ToneLabel.OPTIMISTIC_CAUTIOUS,
            ToneLabel.CONFIDENT_EXPANDING,
        )
        assert direction == SignalDirection.BEARISH
        assert abs(strength - 0.2) < 1e-6  # 1/5 = 0.2

    def test_engine_name(self) -> None:
        assert ToneEngine().name == "tone"
