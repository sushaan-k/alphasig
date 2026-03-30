"""Tests for sigint.parser -- Filing section parser."""

from __future__ import annotations

import pytest

from sigint.exceptions import ParsingError
from sigint.models import Filing, FilingSection, FilingType
from sigint.parser import find_section, parse_filing


class TestParseFiling:
    """Tests for parse_filing()."""

    def test_parses_known_sections(self, sample_filing_with_sections: Filing) -> None:
        sections = parse_filing(sample_filing_with_sections)
        keys = {s.section_key for s in sections}
        assert "risk_factors" in keys
        assert "business" in keys

    def test_sections_contain_text(self, sample_filing_with_sections: Filing) -> None:
        sections = parse_filing(sample_filing_with_sections)
        for section in sections:
            assert len(section.text) > 50

    def test_sections_have_correct_metadata(
        self, sample_filing_with_sections: Filing
    ) -> None:
        sections = parse_filing(sample_filing_with_sections)
        for section in sections:
            assert section.ticker == "AAPL"
            assert section.filing_type == FilingType.TEN_K
            assert section.filing_accession == "0000320193-24-000123"

    def test_raises_on_empty_html(self, sample_filing: Filing) -> None:
        empty = sample_filing.model_copy(update={"raw_html": ""})
        with pytest.raises(ParsingError):
            parse_filing(empty)

    def test_returns_empty_for_no_sections(self, sample_filing: Filing) -> None:
        # Use a long-enough HTML document that passes the minimum size guard
        # but contains no recognised SEC section headers.
        body = "<p>No SEC items here. Just some filler text.</p>" * 15
        plain = sample_filing.model_copy(
            update={"raw_html": f"<html><body>{body}</body></html>"}
        )
        sections = parse_filing(plain)
        assert sections == []


class TestFindSection:
    """Tests for find_section()."""

    def test_finds_existing_section(self, sample_sections: list[FilingSection]) -> None:
        rf = find_section(sample_sections, "risk_factors")
        assert rf is not None
        assert rf.section_key == "risk_factors"

    def test_returns_none_for_missing(
        self, sample_sections: list[FilingSection]
    ) -> None:
        result = find_section(sample_sections, "nonexistent")
        assert result is None
