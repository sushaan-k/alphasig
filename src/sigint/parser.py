"""Filing section parser -- extracts structured text from SEC HTML filings.

SEC filings are messy HTML documents.  This module locates well-known
sections (Risk Factors, MD&A, etc.) by scanning for ``<a name="">`` anchors
and heading patterns, then returns clean plaintext for each section.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import structlog
from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString

from sigint.exceptions import ParsingError
from sigint.models import Filing, FilingSection

logger = structlog.get_logger()

# Canonical section definitions: (key, display_name, pattern).
# Patterns match Item headers in 10-K / 10-Q filings.
_SECTION_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    (
        "risk_factors",
        "Risk Factors",
        re.compile(
            r"item\s+1a[\.\s\u2014\u2013\-]+risk\s+factors",
            re.IGNORECASE,
        ),
    ),
    (
        "md_and_a",
        "Management Discussion and Analysis",
        re.compile(
            r"item\s+(?:7|2)[\.\s\u2014\u2013\-]+"
            r"management.{0,10}discussion",
            re.IGNORECASE,
        ),
    ),
    (
        "business",
        "Business",
        re.compile(
            r"item\s+1[\.\s\u2014\u2013\-]+business(?!\s+overview)",
            re.IGNORECASE,
        ),
    ),
    (
        "financial_statements",
        "Financial Statements",
        re.compile(
            r"item\s+(?:8|1)[\.\s\u2014\u2013\-]+financial\s+statements",
            re.IGNORECASE,
        ),
    ),
    (
        "legal_proceedings",
        "Legal Proceedings",
        re.compile(
            r"item\s+3[\.\s\u2014\u2013\-]+legal\s+proceedings",
            re.IGNORECASE,
        ),
    ),
    (
        "properties",
        "Properties",
        re.compile(
            r"item\s+2[\.\s\u2014\u2013\-]+properties",
            re.IGNORECASE,
        ),
    ),
    (
        "controls_procedures",
        "Controls and Procedures",
        re.compile(
            r"item\s+(?:9a|4)[\.\s\u2014\u2013\-]+controls",
            re.IGNORECASE,
        ),
    ),
    (
        "executive_compensation",
        "Executive Compensation",
        re.compile(
            r"(?:item\s+11|executive\s+compensation)",
            re.IGNORECASE,
        ),
    ),
]


def _extract_text(tag: Tag) -> str:
    """Recursively extract visible text from a BS4 tag, collapsing whitespace."""
    parts: list[str] = []
    for child in tag.descendants:
        if isinstance(child, NavigableString):
            text = child.strip()
            if text:
                parts.append(text)
    raw = " ".join(parts)
    # Collapse runs of whitespace
    return re.sub(r"\s{2,}", " ", raw).strip()


def _find_section_boundaries(
    soup: BeautifulSoup,
) -> list[tuple[str, str, Tag]]:
    """Scan the document for section-header elements.

    Returns a list of ``(section_key, section_name, start_tag)`` tuples
    in document order.
    """
    # Collect all text-bearing tags that might be headers
    candidates: list[tuple[Tag, str]] = []
    for tag in soup.find_all(
        ["b", "strong", "p", "div", "span", "font", "h1", "h2", "h3", "h4"]
    ):
        text = _extract_text(tag)
        if 5 < len(text) < 200:
            candidates.append((tag, text))

    found: list[tuple[str, str, Tag]] = []
    seen_keys: set[str] = set()
    for tag, text in candidates:
        for key, name, pattern in _SECTION_PATTERNS:
            if key in seen_keys:
                continue
            if pattern.search(text):
                found.append((key, name, tag))
                seen_keys.add(key)
                break

    return found


def _text_between(start: Tag, end: Tag | None) -> str:
    """Extract all text between two sibling-ish tags.

    Walks ``next_element`` from *start* until we reach *end* (or the
    document ends), gathering visible text along the way.
    """
    parts: list[str] = []
    node = start.next_element
    while node is not None:
        if node is end:
            break
        if isinstance(node, NavigableString) and not isinstance(node, (type(None),)):
            text = str(node).strip()
            if text:
                parts.append(text)
        node = node.next_element
    raw = " ".join(parts)
    return re.sub(r"\s{2,}", " ", raw).strip()


def parse_filing(filing: Filing) -> list[FilingSection]:
    """Parse a filing's HTML into structured sections.

    Args:
        filing: A :class:`Filing` with ``raw_html`` populated.

    Returns:
        List of :class:`FilingSection` instances for each recognised
        section found in the document.

    Raises:
        ParsingError: If the HTML cannot be parsed at all.
    """
    if not filing.raw_html:
        raise ParsingError(f"Filing {filing.accession_number} has no raw_html to parse")

    try:
        soup = BeautifulSoup(filing.raw_html, "lxml")
    except Exception as exc:
        raise ParsingError(
            f"Failed to parse HTML for {filing.accession_number}"
        ) from exc

    boundaries = _find_section_boundaries(soup)
    if not boundaries:
        logger.warning(
            "no_sections_found",
            accession=filing.accession_number,
            ticker=filing.ticker,
        )
        return []

    sections: list[FilingSection] = []
    for idx, (key, name, start_tag) in enumerate(boundaries):
        end_tag = boundaries[idx + 1][2] if idx + 1 < len(boundaries) else None
        text = _text_between(start_tag, end_tag)

        # Skip trivially short sections (likely parsing artefacts)
        if len(text) < 100:
            continue

        sections.append(
            FilingSection(
                filing_accession=filing.accession_number,
                ticker=filing.ticker,
                section_name=name,
                section_key=key,
                text=text,
                filing_type=filing.filing_type,
                filed_date=filing.filed_date,
            )
        )

    logger.info(
        "filing_parsed",
        accession=filing.accession_number,
        ticker=filing.ticker,
        sections=[s.section_key for s in sections],
    )
    return sections


def find_section(sections: Sequence[FilingSection], key: str) -> FilingSection | None:
    """Find a section by its normalised key.

    Args:
        sections: Parsed sections from :func:`parse_filing`.
        key: Section key such as ``"risk_factors"`` or ``"md_and_a"``.

    Returns:
        The matching section, or ``None``.
    """
    for s in sections:
        if s.section_key == key:
            return s
    return None
