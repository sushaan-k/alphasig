"""Tests for alphasig.parser -- Filing section parser."""

from __future__ import annotations

import pytest

from alphasig.exceptions import ParsingError
from alphasig.models import Filing, FilingSection, FilingType
from alphasig.parser import find_section, parse_filing


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


def _filing(html: str, filing_type: FilingType = FilingType.TEN_K) -> Filing:
    from datetime import date

    return Filing(
        accession_number="0000000000-24-000001",
        cik="0000000001",
        ticker="TEST",
        company_name="Test Co",
        filing_type=filing_type,
        filed_date=date(2024, 2, 1),
        period_of_report=date(2023, 12, 31),
        url="https://www.sec.gov/Archives/edgar/data/1/000000000024000001/t.htm",
        raw_html=f"<html><body>{html}</body></html>",
    )


_FILLER = "Material risk language that is long enough to be a real paragraph. " * 5


class TestSectionBoundaries:
    """Regression tests for real-world 10-K / 10-Q layouts."""

    def test_table_of_contents_does_not_hide_sections(self) -> None:
        toc = "".join(
            f"<p>{h}</p><p>{page}</p>"
            for h, page in [
                ("Item 1. Business", 3),
                ("Item 1A. Risk Factors", 10),
                ("Item 7. Management's Discussion and Analysis", 30),
                ("Item 11. Executive Compensation", 90),
            ]
        )
        body = (
            f"<div>{toc}</div>"
            f"<p><b>Item 1. Business</b></p><p>BUSINESS {_FILLER}</p>"
            f"<p><b>Item 1A. Risk Factors</b></p><p>RISKS {_FILLER}</p>"
            f"<p><b>Item 7. Management's Discussion and Analysis</b></p>"
            f"<p>MDA {_FILLER}</p>"
            f"<p><b>Item 11. Executive Compensation</b></p><p>COMP {_FILLER}</p>"
        )
        sections = {s.section_key: s.text for s in parse_filing(_filing(body))}
        assert {"business", "risk_factors", "md_and_a"} <= sections.keys()
        assert "RISKS" in sections["risk_factors"]
        assert "MDA" not in sections["risk_factors"]
        assert "BUSINESS" not in sections["executive_compensation"]

    def test_risk_factors_stop_at_unextracted_items(self) -> None:
        body = (
            f"<p><b>Item 1A. Risk Factors</b></p><p>RISKS {_FILLER}</p>"
            f"<p><b>Item 1B. Unresolved Staff Comments</b></p><p>None. {_FILLER}</p>"
            f"<p><b>Item 1C. Cybersecurity</b></p><p>CYBER {_FILLER}</p>"
            f"<p><b>Item 2. Properties</b></p><p>PROPS {_FILLER}</p>"
        )
        rf = find_section(parse_filing(_filing(body)), "risk_factors")
        assert rf is not None
        assert "CYBER" not in rf.text
        assert "Unresolved" not in rf.text

    @pytest.mark.parametrize(
        "split", ["<span>R</span><span>isk</span>", "Ri<span>sk</span>"]
    )
    def test_heading_with_separately_styled_initial(self, split: str) -> None:
        """Regression: Oracle styles each word's first letter(s) separately,
        which flattens to "R isk Factors" / "Ri sk Factors"."""
        body = (
            f"<p><b>Item 1. Business</b></p><p>BUSINESS {_FILLER}</p>"
            f"<p><b>Item 1A. {split} Factors</b></p><p>RISKS {_FILLER}</p>"
            f"<p><b>Item 2. Properties</b></p><p>PROPS {_FILLER}</p>"
        )
        rf = find_section(parse_filing(_filing(body)), "risk_factors")
        assert rf is not None
        assert "RISKS" in rf.text
        assert "PROPS" not in rf.text

    def test_ten_q_part_ii_risk_factors(self) -> None:
        body = (
            f"<p>PART I, Item 2. Management's Discussion and Analysis</p>"
            f"<p>MDA {_FILLER}</p>"
            f"<p>PART II — Item 1A. Risk Factors</p><p>RISKS {_FILLER}</p>"
            f"<p>Item 2. Unregistered Sales of Equity Securities</p>"
            f"<p>BUYBACKS {_FILLER}</p><p>Item 6. Exhibits</p><p>EXHIBITS {_FILLER}</p>"
        )
        sections = parse_filing(_filing(body, FilingType.TEN_Q))
        rf = find_section(sections, "risk_factors")
        assert rf is not None
        assert "RISKS" in rf.text
        assert "BUYBACKS" not in rf.text
        assert find_section(sections, "md_and_a") is not None

    def test_cross_reference_is_not_a_header(self) -> None:
        body = (
            f"<p><b>Item 7. Management's Discussion and Analysis</b></p>"
            f"<p>See Item 1A. Risk Factors.</p><p>MDA {_FILLER * 2}</p>"
        )
        sections = parse_filing(_filing(body))
        assert find_section(sections, "risk_factors") is None
        mda = find_section(sections, "md_and_a")
        assert mda is not None and "MDA" in mda.text

    def test_eight_k_returns_whole_report(self) -> None:
        body = (
            "<p>Item 1.01 Entry into a Material Definitive Agreement</p>"
            f"<p>MERGER {_FILLER * 2}</p><p>Item 9.01 Financial Statements</p>"
        )
        sections = parse_filing(_filing(body, FilingType.EIGHT_K))
        assert [s.section_key for s in sections] == ["current_report"]
        assert "MERGER" in sections[0].text


def test_running_page_headers_do_not_split_a_section() -> None:
    body = (
        f"<p><b>Item 1A. Risk Factors</b></p><p>PAGE1 {_FILLER * 3}</p>"
        f"<p><b>Item 1A. Risk Factors (continued)</b></p><p>PAGE2 {_FILLER}</p>"
        f"<p><b>Item 1B. Unresolved Staff Comments</b></p><p>None. {_FILLER}</p>"
    )
    rf = find_section(parse_filing(_filing(body)), "risk_factors")
    assert rf is not None
    assert "PAGE1" in rf.text
    assert "PAGE2" in rf.text
    assert "Unresolved" not in rf.text


def test_sections_carry_acceptance_time() -> None:
    from datetime import UTC, datetime

    filing = _filing(
        f"<p><b>Item 1A. Risk Factors</b></p><p>{_FILLER * 3}</p>"
    ).model_copy(update={"accepted_at": datetime(2024, 2, 1, 21, 5, tzinfo=UTC)})
    (section,) = parse_filing(filing)
    assert section.available_at == datetime(2024, 2, 1, 21, 5, tzinfo=UTC)
