"""Filing section parser -- extracts structured text from SEC HTML filings.

SEC filings are messy HTML documents.  This module locates well-known
sections (Risk Factors, MD&A, etc.) by scanning for "Item N." heading
elements, then returns clean plaintext for each section.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import structlog
from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString

from alphasig.exceptions import ParsingError
from alphasig.models import Filing, FilingSection, FilingType

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
            r"(?:item\s+11\b|executive\s+compensation\s*$)",
            re.IGNORECASE,
        ),
    ),
]


# Any "Item N." / "Item 1A." / "Item 2.02" header -- used to terminate the
# preceding section even when the next item is not one we extract (e.g.
# Item 1B/1C after Risk Factors, or 10-Q Part II Item 2).
_ITEM_HEADER = re.compile(r"item\s+\d{1,2}[a-c]?(?:\.\d{2})?\b", re.IGNORECASE)
_PART_PREFIX = re.compile(r"part\s+[iv]+\W+", re.IGNORECASE)
_HEADER_TAGS = ["b", "strong", "p", "div", "span", "font", "h1", "h2", "h3", "h4"]
_MAX_HEADER_LEN = 200


def _header_text(tag: Tag) -> str | None:
    """Return the tag's collapsed text, or ``None`` if too long for a header.

    Stops walking as soon as the text exceeds the header limit, so large
    container elements cost O(limit) instead of O(document).
    """
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
    """Scan the document for Item header elements, in document order.

    Returns ``(section_key, section_name, tag)`` tuples; key and name are
    ``None`` for Item headers we do not extract, which still act as the end
    boundary of the preceding section.  Headers must *start* with their
    Item label so cross-references in running text are not mistaken for
    headers.
    """
    headers: list[tuple[str | None, str | None, Tag]] = []
    for tag in soup.find_all(_HEADER_TAGS):
        # Nested markup (<p><b>Item 1A...</b></p>) repeats the same header.
        # (Identity check: bs4's Tag equality compares whole subtrees.)
        if headers and any(p is headers[-1][2] for p in tag.parents):
            continue
        text = _header_text(tag)
        if text is None or len(text) <= 5:
            continue
        prefix = _PART_PREFIX.match(text)
        if prefix:
            text = text[prefix.end() :]
        match = next(
            ((key, name) for key, name, pat in _SECTION_PATTERNS if pat.match(text)),
            None,
        )
        if match is not None:
            # A repeated running header ("Item 1A. Risk Factors (continued)")
            # continues the section instead of splitting it.
            if headers and headers[-1][0] == match[0]:
                continue
            headers.append((match[0], match[1], tag))
        elif _ITEM_HEADER.match(text):
            headers.append((None, None, tag))
    return headers


def _text_between(start: Tag, end: Tag | None) -> str:
    """Extract all visible text from *start* up to (not including) *end*."""
    parts: list[str] = []
    node = start.next_element
    while node is not None and node is not end:
        if isinstance(node, NavigableString):
            text = node.strip()
            if text:
                parts.append(text)
        node = node.next_element
    return re.sub(r"\s{2,}", " ", " ".join(parts)).strip()


def _make_section(filing: Filing, key: str, name: str, text: str) -> FilingSection:
    return FilingSection(
        filing_accession=filing.accession_number,
        ticker=filing.ticker,
        section_name=name,
        section_key=key,
        text=text,
        filing_type=filing.filing_type,
        filed_date=filing.filed_date,
        accepted_at=filing.accepted_at,
    )


def parse_filing(filing: Filing) -> list[FilingSection]:
    """Parse a filing's HTML into structured sections.

    For 10-K / 10-Q style documents each recognised Item runs until the
    next Item header of any kind.  A heading repeated back-to-back (running
    page headers) continues the section; when it reappears elsewhere (table
    of contents), the occurrence with the longest body wins.  8-K current reports are short and event-driven, so the whole
    document is returned as a single ``current_report`` section.

    Args:
        filing: A :class:`Filing` with ``raw_html`` populated.

    Returns:
        List of :class:`FilingSection` instances, in document order.

    Raises:
        ParsingError: If the HTML cannot be parsed at all.
    """
    if not filing.raw_html:
        raise ParsingError(f"Filing {filing.accession_number} has no raw_html to parse")
    if len(filing.raw_html) < 500:
        raise ParsingError(
            f"Filing {filing.accession_number} raw_html is too short "
            f"({len(filing.raw_html)} chars) — likely truncated or empty"
        )

    try:
        soup = BeautifulSoup(filing.raw_html, "lxml")
    except Exception as exc:
        raise ParsingError(
            f"Failed to parse HTML for {filing.accession_number}"
        ) from exc

    if filing.filing_type is FilingType.EIGHT_K:
        root = soup.body or soup
        text = re.sub(r"\s+", " ", root.get_text(" ")).strip()
        return [_make_section(filing, "current_report", "Current Report", text)]

    boundaries = _find_section_boundaries(soup)
    best: dict[str, tuple[int, str, str]] = {}  # key -> (position, name, text)
    for idx, (key, name, start_tag) in enumerate(boundaries):
        if key is None or name is None:
            continue
        end_tag = boundaries[idx + 1][2] if idx + 1 < len(boundaries) else None
        text = _text_between(start_tag, end_tag)
        if key not in best or len(text) > len(best[key][2]):
            best[key] = (idx, name, text)

    sections = [
        _make_section(filing, key, name, text)
        for key, (_, name, text) in sorted(best.items(), key=lambda kv: kv[1][0])
        # Skip trivially short sections (likely parsing artefacts)
        if len(text) >= 100
    ]
    if not sections:
        logger.warning(
            "no_sections_found",
            accession=filing.accession_number,
            ticker=filing.ticker,
        )
        return []

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
