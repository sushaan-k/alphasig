"""Filing section parser -- extracts structured text from SEC HTML filings.

SEC filings are messy HTML documents.  This module locates well-known
sections (Risk Factors, MD&A, etc.) by scanning for "Item N." heading
elements, then returns clean plaintext for each section.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import structlog
from bs4 import BeautifulSoup
from lxml import etree

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
# Headings that style a word's first letter(s) separately
# ("<span>R</span>isk Factors") flatten to "R isk Factors" or
# "Ri sk Factors"; rejoin the fragment before matching.
_SPLIT_INITIAL = re.compile(r"\b([A-Z][a-z]?) (?=[a-z]{2,})")
_HEADER_TAGS = frozenset(
    {"b", "strong", "p", "div", "span", "font", "h1", "h2", "h3", "h4"}
)
_MAX_HEADER_LEN = 200
_WS = re.compile(r"\s+")
_WS_RUN = re.compile(r"\s{2,}")


class _FlatDocument:
    """A filing flattened once into its text nodes, in document order.

    ``chunks`` holds every non-blank text node (element text, tail text,
    comment and processing-instruction text), stripped -- the same strings,
    in the same order, as a walk over the document's ``NavigableString``
    nodes.  Each candidate header element is recorded as ``(first_chunk,
    end_chunk, start_event, end_event)``: its text is ``chunks[first:end]``
    and the text between two headers is a slice from one ``first_chunk`` to
    the next, so the document is walked once instead of once per tag.
    """

    __slots__ = ("candidates", "chunks", "prefix")

    def __init__(self, root: etree._Element) -> None:
        chunks: list[str] = []
        # prefix[i] == sum(len(c) + 1 for c in chunks[:i])
        prefix = [0]
        candidates: list[list[int]] = []
        open_stack: list[list[int]] = []

        def add(raw: str | None) -> None:
            if raw:
                text = raw.strip()
                if text:
                    chunks.append(text)
                    prefix.append(prefix[-1] + len(text) + 1)

        events = etree.iterwalk(root, events=("start", "end", "comment", "pi"))
        for event_no, (event, node) in enumerate(events):
            if event == "start":
                if node.tag in _HEADER_TAGS:
                    record = [len(chunks), -1, event_no, -1]
                    candidates.append(record)
                    open_stack.append(record)
                add(node.text)
            elif event == "end":
                if node.tag in _HEADER_TAGS:
                    record = open_stack.pop()
                    record[1] = len(chunks)
                    record[3] = event_no
                add(node.tail)
            else:  # comment / processing instruction
                add(node.text)
                add(node.tail)
        # Comments or processing instructions after the root element.
        for sibling in root.itersiblings():
            add(sibling.text)
            add(sibling.tail)

        self.chunks = chunks
        self.prefix = prefix
        self.candidates = candidates

    def header_text(self, first: int, end: int) -> str | None:
        """Collapsed text of ``chunks[first:end]``, or ``None`` if too long."""
        if self.prefix[end] - self.prefix[first] > _MAX_HEADER_LEN * 2:
            return None
        text = _WS.sub(" ", " ".join(self.chunks[first:end])).strip()
        return text if len(text) < _MAX_HEADER_LEN else None

    def text(self, first: int, end: int | None) -> str:
        """Visible text from chunk *first* up to (not including) *end*."""
        return _WS_RUN.sub(" ", " ".join(self.chunks[first:end])).strip()


def _parse_html(html: str) -> etree._Element | None:
    """Parse *html* with libxml2's forgiving HTML parser.

    The feed interface also accepts ``str`` input that starts with an XML
    encoding declaration, as inline-XBRL filings do.
    """
    parser = etree.HTMLParser(recover=True)
    parser.feed(html)
    root: etree._Element | None = parser.close()
    return root


def _find_section_boundaries(
    doc: _FlatDocument,
) -> list[tuple[str | None, str | None, int]]:
    """Scan the document for Item header elements, in document order.

    Returns ``(section_key, section_name, first_chunk)`` tuples; key and
    name are ``None`` for Item headers we do not extract, which still act
    as the end boundary of the preceding section.  Headers must *start*
    with their Item label so cross-references in running text are not
    mistaken for headers.
    """
    headers: list[tuple[str | None, str | None, int]] = []
    last_end_event = -1
    for first, end, start_event, end_event in doc.candidates:
        # Nested markup (<p><b>Item 1A...</b></p>) repeats the same header.
        if start_event < last_end_event:
            continue
        text = doc.header_text(first, end)
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
            # A repeated running header ("Item 1A. Risk Factors (continued)")
            # continues the section instead of splitting it.
            if headers and headers[-1][0] == match[0]:
                continue
            headers.append((match[0], match[1], first))
            last_end_event = end_event
        elif _ITEM_HEADER.match(text):
            headers.append((None, None, first))
            last_end_event = end_event
    return headers


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

    if filing.filing_type is FilingType.EIGHT_K:
        try:
            soup = BeautifulSoup(filing.raw_html, "lxml")
        except Exception as exc:
            raise ParsingError(
                f"Failed to parse HTML for {filing.accession_number}"
            ) from exc
        root = soup.body or soup
        text = re.sub(r"\s+", " ", root.get_text(" ")).strip()
        return [_make_section(filing, "current_report", "Current Report", text)]

    try:
        tree = _parse_html(filing.raw_html)
    except Exception as exc:
        raise ParsingError(
            f"Failed to parse HTML for {filing.accession_number}"
        ) from exc

    doc = _FlatDocument(tree) if tree is not None else None
    boundaries = _find_section_boundaries(doc) if doc is not None else []
    best: dict[str, tuple[int, str, str]] = {}  # key -> (position, name, text)
    for idx, (key, name, first) in enumerate(boundaries):
        if key is None or name is None or doc is None:
            continue
        end = boundaries[idx + 1][2] if idx + 1 < len(boundaries) else None
        text = doc.text(first, end)
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
