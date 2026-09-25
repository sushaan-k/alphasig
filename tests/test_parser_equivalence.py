"""The single-pass lxml section scan must reproduce the BeautifulSoup
implementation it replaced exactly (same sections, same text).

The oracle below is that implementation (``alphasig.parser`` as of 0.2.0's
parser rewrite), kept verbatim: a BeautifulSoup(lxml) tree, per-tag
descendant text for header candidates, and a ``next_element`` walk between
headers.  8-K filings still go through BeautifulSoup and are not covered.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest
from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString

from alphasig.models import Filing, FilingType
from alphasig.parser import _SECTION_PATTERNS, parse_filing
from benchmarks.fixtures import CACHE_DIR, REAL_FIXTURES, VARIANTS, synth_filing

# Inline-XBRL filings start with an XML declaration; bs4 warns about that,
# which is irrelevant to the oracle.
pytestmark = pytest.mark.filterwarnings("ignore::bs4.XMLParsedAsHTMLWarning")

# --- oracle: the BeautifulSoup implementation, verbatim -------------------
# (plus the split-initial heading normalisation both implementations share)

_ITEM_HEADER = re.compile(r"item\s+\d{1,2}[a-c]?(?:\.\d{2})?\b", re.IGNORECASE)
_PART_PREFIX = re.compile(r"part\s+[iv]+\W+", re.IGNORECASE)
_SPLIT_INITIAL = re.compile(r"\b([A-Z][a-z]?) (?=[a-z]{2,})")
_HEADER_TAGS = ["b", "strong", "p", "div", "span", "font", "h1", "h2", "h3", "h4"]
_MAX_HEADER_LEN = 200


def _header_text(tag: Tag) -> str | None:
    parts: list[str] = []
    length = 0
    for child in tag.descendants:
        if isinstance(child, NavigableString):
            text = child.strip()
            if text:
                parts.append(text)
                length += len(text) + 1
                if length > _MAX_HEADER_LEN * 2:
                    return None
    text = re.sub(r"\s+", " ", " ".join(parts)).strip()
    return text if len(text) < _MAX_HEADER_LEN else None


def _find_section_boundaries(
    soup: BeautifulSoup,
) -> list[tuple[str | None, str | None, Tag]]:
    headers: list[tuple[str | None, str | None, Tag]] = []
    for tag in soup.find_all(_HEADER_TAGS):
        if headers and any(p is headers[-1][2] for p in tag.parents):
            continue
        text = _header_text(tag)
        if text is None or len(text) <= 5:
            continue
        prefix = _PART_PREFIX.match(text)
        if prefix:
            text = text[prefix.end() :]
        text = _SPLIT_INITIAL.sub(r"\1", text)
        match = next(
            ((key, name) for key, name, pat in _SECTION_PATTERNS if pat.match(text)),
            None,
        )
        if match is not None:
            if headers and headers[-1][0] == match[0]:
                continue
            headers.append((match[0], match[1], tag))
        elif _ITEM_HEADER.match(text):
            headers.append((None, None, tag))
    return headers


def _text_between(start: Tag, end: Tag | None) -> str:
    parts: list[str] = []
    node = start.next_element
    while node is not None and node is not end:
        if isinstance(node, NavigableString):
            text = node.strip()
            if text:
                parts.append(text)
        node = node.next_element
    return re.sub(r"\s{2,}", " ", " ".join(parts)).strip()


def _oracle(html: str) -> list[tuple[str, str, str]]:
    soup = BeautifulSoup(html, "lxml")
    boundaries = _find_section_boundaries(soup)
    best: dict[str, tuple[int, str, str]] = {}
    for idx, (key, name, start_tag) in enumerate(boundaries):
        if key is None or name is None:
            continue
        end_tag = boundaries[idx + 1][2] if idx + 1 < len(boundaries) else None
        text = _text_between(start_tag, end_tag)
        if key not in best or len(text) > len(best[key][2]):
            best[key] = (idx, name, text)
    return [
        (key, name, text)
        for key, (_, name, text) in sorted(best.items(), key=lambda kv: kv[1][0])
        if len(text) >= 100
    ]


# --- comparison -----------------------------------------------------------


def _filing(html: str, form: str = "10-K") -> Filing:
    return Filing(
        accession_number="0000000000-24-000001",
        cik="0000000000",
        ticker="EQ",
        company_name="Equivalence Corp",
        filing_type=FilingType(form),
        filed_date=date(2024, 1, 1),
        period_of_report=date(2023, 12, 31),
        url="https://example.invalid/eq.htm",
        raw_html=html,
    )


def _ours(html: str, form: str = "10-K") -> list[tuple[str, str, str]]:
    return [
        (s.section_key, s.section_name, s.text)
        for s in parse_filing(_filing(html, form))
    ]


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("form", ["10-K", "10-Q"])
def test_matches_bs4_on_synthetic_filings(variant: str, form: str) -> None:
    html = synth_filing(7, form=form, variant=variant, tables=2).html
    expected = _oracle(html)
    assert expected  # the fixture exercises the section scan
    assert _ours(html, form) == expected


def test_matches_bs4_on_messy_markup() -> None:
    html = (
        "<!DOCTYPE html><!-- lead comment --><html><body>"
        "<p><b>Item 1. Business</b><!-- inline comment --></p>"
        + "<div>We make <i>widgets</i>&#160;and&nbsp;gadgets.\n\n   More   text.</div>"
        * 20
        + "<table><tr><td><font>Item 1A.</font></td><td>Risk Factors</td></tr></table>"
        + "<p><strong>PART I — ITEM 1A — RISK FACTORS</strong></p>"
        + "<p>Risk <span>nested <b>deep</b></span> text, unclosed <b>bold" * 20
        + "<p><b>Item 1A. Risk Factors (continued)</b></p><p>More risk.</p>"
        + "<div><span>Item 1B.</span><span>\tUnresolved\nStaff Comments</span></div>"
        + "<p>None.</p>"
        # Too long to be a header, even after whitespace collapses.
        + f"<p>{'x ' * 300}</p>"
        + f"<p>Item 2. Properties{' ' * 500}tail</p>"
        + "<h2>Item 7. Management\u2019s Discussion and Analysis</h2>"
        + "<p>Revenue grew.</p><script>var x = 1;</script>" * 20
        + "</body></html><!-- trailing comment -->"
    )
    expected = _oracle(html)
    assert [key for key, _, _ in expected] == ["business", "risk_factors", "md_and_a"]
    assert _ours(html) == expected


@pytest.mark.parametrize("fx", REAL_FIXTURES, ids=lambda f: f.name)
def test_matches_bs4_on_cached_real_filings(fx) -> None:  # type: ignore[no-untyped-def]
    path = Path(CACHE_DIR) / f"{fx.name}.html"
    if not path.exists():
        pytest.skip("real fixture not cached (run python -m benchmarks.run once)")
    html = path.read_text(encoding="utf-8")
    assert _ours(html, fx.form) == _oracle(html)
