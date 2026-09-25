"""Benchmark fixtures: pinned real filings plus a seeded synthetic generator.

Real fixtures
    Five public SEC filings (Apple, NVIDIA and Oracle 10-K / 10-Q primary
    documents) that the MIT-licensed ``dgunning/edgartools`` project ships
    as test data.  They are fetched from ``raw.githubusercontent.com`` at the
    pinned tag below, verified against SHA-256 checksums, and cached under
    ``benchmarks/.cache/`` (git-ignored).  sec.gov itself is never contacted.
    If they cannot be fetched (offline CI), benches fall back to synthetic
    filings only and record that fact in the result JSON.

Synthetic fixtures
    :func:`synth_filing` builds Workiva-style inline-XBRL-like HTML (nested
    ``<div><span style=...>`` markup, a table of contents, financial tables,
    anchor ids on every Item heading) from a seeded RNG, so output is
    byte-for-byte reproducible.  Anchor ids give exact ground truth for the
    section-boundary benchmark.
"""

from __future__ import annotations

import hashlib
import random
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent / ".cache"

_EDGARTOOLS_TAG = "v5.58.0"
_BASE = (
    f"https://raw.githubusercontent.com/dgunning/edgartools/{_EDGARTOOLS_TAG}/data/html"
)


@dataclass(frozen=True)
class RealFixture:
    name: str
    form: str
    sha256: str
    # section_key -> (start anchor id, end anchor id); labels were set by
    # manual inspection of each document's table-of-contents hyperlinks.
    gold: dict[str, tuple[str, str]] = field(default_factory=dict)

    @property
    def url(self) -> str:
        return f"{_BASE}/{self.name}.html"


REAL_FIXTURES: list[RealFixture] = [
    RealFixture(
        "Apple.10-K",
        "10-K",
        "ba4222c4fbd8ddfbd63982bcef63ffefba232bf079a89d628418be5f10935af0",
        {
            "risk_factors": (
                "i7bfbfbe54b9647b1b4ba4ff4e0aba09d_52",
                "i7bfbfbe54b9647b1b4ba4ff4e0aba09d_70",
            ),
            "md_and_a": (
                "i7bfbfbe54b9647b1b4ba4ff4e0aba09d_94",
                "i7bfbfbe54b9647b1b4ba4ff4e0aba09d_166",
            ),
        },
    ),
    RealFixture(
        "Apple.10-Q",
        "10-Q",
        "3523413093a5d3f2e72b4fdc11cb7268c26e12c54f5ccc33e557dde9d2feb0c2",
        {
            "md_and_a": (
                "i399fc64bbe494642a709f5ddee803e6a_67",
                "i399fc64bbe494642a709f5ddee803e6a_148",
            ),
            "risk_factors": (
                "i399fc64bbe494642a709f5ddee803e6a_160",
                "i399fc64bbe494642a709f5ddee803e6a_163",
            ),
        },
    ),
    RealFixture(
        "Nvidia.10-K",
        "10-K",
        "e87d82649d5241850776122abf604ea7c731f78d974a22323cace913af4dc836",
        {
            "risk_factors": (
                "i8ce5c25b938445b1bec835777d6cece9_16",
                "i8ce5c25b938445b1bec835777d6cece9_19",
            ),
            "md_and_a": (
                "i8ce5c25b938445b1bec835777d6cece9_40",
                "i8ce5c25b938445b1bec835777d6cece9_52",
            ),
        },
    ),
    RealFixture(
        "Oracle.10-K",
        "10-K",
        "4c7ede9e1c3c12a2c4fe7b246b10408c1332eb3016c8dbd10a569a25f0462133",
        {
            "risk_factors": (
                "item_1a_risk_factors",
                "item_1b_unresolved_staff_comments",
            ),
            "md_and_a": (
                "item_7_managements_discussion_analysis",
                "item_7a_quantitative_qualitative_disclos",
            ),
        },
    ),
    RealFixture(
        "Oracle.10-Q",
        "10-Q",
        "08f0b779e82b5127b9a7fb9901ac2152deddfe3af25bfa80333f70c2bb4602ac",
        {
            "md_and_a": (
                "item_2_managements_discussion_analysis_f",
                "item_3_quantitative_qualitative_disclosu",
            ),
            "risk_factors": (
                "item_1a_risk_factors",
                "item_2_unregistered_sales_equity_securit",
            ),
        },
    ),
]


def load_real_fixtures(*, allow_download: bool = True) -> list[tuple[RealFixture, str]]:
    """Return ``(fixture, html)`` pairs for every verifiable real fixture.

    Files failing checksum verification are discarded, never used.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out: list[tuple[RealFixture, str]] = []
    for fx in REAL_FIXTURES:
        path = CACHE_DIR / f"{fx.name}.html"
        data: bytes | None = path.read_bytes() if path.exists() else None
        if data is None and allow_download:
            try:
                with urllib.request.urlopen(fx.url, timeout=60) as resp:
                    data = resp.read()
            except OSError:
                data = None
        if data is None or hashlib.sha256(data).hexdigest() != fx.sha256:
            continue
        if not path.exists():
            path.write_bytes(data)
        out.append((fx, data.decode("utf-8")))
    return out


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------

_SUPPLIERS = [
    "Taiwan Semiconductor Manufacturing Company",
    "Samsung Electronics",
    "Foxconn",
    "Micron Technology",
    "SK Hynix",
    "Broadcom",
    "Amkor Technology",
    "ASML",
    "Corning",
    "Murata Manufacturing",
]
_TOPICS = [
    "supply chain concentration",
    "export controls",
    "data privacy regulation",
    "cybersecurity incidents",
    "foreign exchange volatility",
    "antitrust enforcement",
    "component shortages",
    "geopolitical tensions in the Asia-Pacific region",
    "climate-related disclosure requirements",
    "interest rate changes",
    "customer concentration",
    "intellectual property litigation",
    "labor availability",
    "tax law changes",
    "artificial intelligence regulation",
]
_VERBS_SOFT = ["may", "could", "might"]
_VERBS_HARD = ["is currently", "has been", "continues to be"]
_FILLER = (
    "results of operations",
    "financial condition",
    "gross margin",
    "net sales",
    "operating expenses",
    "cash flows",
    "future periods",
    "the Company's products and services",
    "demand for the Company's products",
    "third-party service providers",
)

_LINE_ITEMS = (
    "Net sales",
    "Cost of sales",
    "Gross margin",
    "Research and development",
    "Selling, general and administrative",
    "Operating income",
    "Other income/(expense), net",
    "Income before provision for income taxes",
    "Provision for income taxes",
    "Net income",
    "Cash and cash equivalents",
    "Accounts receivable, net",
    "Inventories",
    "Property, plant and equipment, net",
    "Total assets",
    "Accounts payable",
    "Deferred revenue",
    "Term debt",
    "Total liabilities",
    "Total shareholders' equity",
)

_SPAN = (
    "<span style=\"color:#000000;font-family:'Helvetica',sans-serif;"
    'font-size:9pt;font-weight:{w};line-height:120%">{t}</span>'
)


def _p(text: str, *, weight: int = 400) -> str:
    return (
        '<div style="margin-top:9pt;text-align:justify">'
        + _SPAN.format(w=weight, t=text)
        + "</div>\n"
    )


def _sentence(rng: random.Random, topic: str, hard: bool = False) -> str:
    verb = rng.choice(_VERBS_HARD if hard else _VERBS_SOFT)
    return (
        f"The Company {verb} be adversely affected by {topic}, which "
        f"{rng.choice(_VERBS_SOFT)} materially affect its "
        f"{rng.choice(_FILLER)} and {rng.choice(_FILLER)}. "
        f"In particular, reliance on {rng.choice(_SUPPLIERS)} for "
        f"{rng.choice(['wafer fabrication', 'final assembly', 'memory', 'packaging', 'displays'])} "
        f"creates exposure to disruption in {rng.randint(2019, 2026)}."
    )


def _para(rng: random.Random, n: int, topic: str | None = None) -> str:
    return " ".join(
        _sentence(rng, topic or rng.choice(_TOPICS), hard=rng.random() < 0.2)
        for _ in range(n)
    )


def _table(rng: random.Random, rows: int, cols: int = 4) -> str:
    out = ['<table style="border-collapse:collapse;width:100%">']
    for r in range(rows):
        cells = [
            f'<td style="padding:2px"><div>{_SPAN.format(w=400, t=_LINE_ITEMS[r % len(_LINE_ITEMS)])}</div></td>'
        ]
        for _ in range(cols):
            val = f"{rng.randint(100, 999_999):,}"
            cells.append(
                '<td style="padding:2px;text-align:right"><div>'
                + _SPAN.format(w=400, t="$")
                + "</div></td>"
                + '<td style="padding:2px;text-align:right"><div>'
                + _SPAN.format(w=400, t=val)
                + "</div></td>"
            )
        out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</table>\n")
    return "".join(out)


# (anchor id, item label, title) in document order
_ITEMS_10K = [
    ("item1", "Item 1.", "Business"),
    ("item1a", "Item 1A.", "Risk Factors"),
    ("item1b", "Item 1B.", "Unresolved Staff Comments"),
    ("item1c", "Item 1C.", "Cybersecurity"),
    ("item2", "Item 2.", "Properties"),
    ("item3", "Item 3.", "Legal Proceedings"),
    ("item4", "Item 4.", "Mine Safety Disclosures"),
    ("item5", "Item 5.", "Market for Registrant's Common Equity"),
    ("item6", "Item 6.", "[Reserved]"),
    (
        "item7",
        "Item 7.",
        "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    ),
    (
        "item7a",
        "Item 7A.",
        "Quantitative and Qualitative Disclosures About Market Risk",
    ),
    ("item8", "Item 8.", "Financial Statements and Supplementary Data"),
    ("item9", "Item 9.", "Changes in and Disagreements with Accountants"),
    ("item9a", "Item 9A.", "Controls and Procedures"),
    ("item10", "Item 10.", "Directors, Executive Officers and Corporate Governance"),
    ("item11", "Item 11.", "Executive Compensation"),
    ("item15", "Item 15.", "Exhibit and Financial Statement Schedules"),
]
_ITEMS_10Q = [
    ("p1item1", "Item 1.", "Financial Statements"),
    (
        "p1item2",
        "Item 2.",
        "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    ),
    (
        "p1item3",
        "Item 3.",
        "Quantitative and Qualitative Disclosures About Market Risk",
    ),
    ("p1item4", "Item 4.", "Controls and Procedures"),
    ("p2item1", "Item 1.", "Legal Proceedings"),
    ("p2item1a", "Item 1A.", "Risk Factors"),
    (
        "p2item2",
        "Item 2.",
        "Unregistered Sales of Equity Securities and Use of Proceeds",
    ),
    ("p2item6", "Item 6.", "Exhibits"),
]
GOLD_10K = {"risk_factors": ("item1a", "item1b"), "md_and_a": ("item7", "item7a")}
GOLD_10Q = {"risk_factors": ("p2item1a", "p2item2"), "md_and_a": ("p1item2", "p1item3")}

# Heading / TOC styles.  "workiva" mirrors the real fixtures (split TOC cells,
# anchor div before a bold heading span).  The others are common variants.
VARIANTS = ("workiva", "inline_toc", "caps_headings", "split_heading", "no_toc")


@dataclass
class SynthFiling:
    html: str
    form: str
    variant: str
    gold: dict[str, tuple[str, str]]
    risk_paragraphs: list[str]


def risk_paragraphs(rng: random.Random, n: int) -> list[str]:
    """A list of risk-factor paragraphs (one topic each)."""
    return [_para(rng, rng.randint(4, 8), topic=rng.choice(_TOPICS)) for _ in range(n)]


def evolve_risks(
    rng: random.Random, paras: list[str], edit_frac: float = 0.1
) -> list[str]:
    """Next-year risk factors: escalate some language, drop one, add one."""
    out = []
    for para in paras:
        if rng.random() < edit_frac:
            para = para.replace(" may ", " is currently ", 1)
        out.append(para)
    if len(out) > 3:
        out.pop(rng.randrange(len(out)))
    out.insert(rng.randrange(len(out) + 1), _para(rng, 5, topic=rng.choice(_TOPICS)))
    return out


def synth_filing(
    seed: int,
    *,
    form: str = "10-K",
    variant: str = "workiva",
    company: str = "Example Corp",
    body_paragraphs: int = 12,
    table_rows: int = 40,
    tables: int = 8,
    risks: list[str] | None = None,
) -> SynthFiling:
    """Generate one synthetic filing deterministically from *seed*."""
    rng = random.Random(seed)
    items = _ITEMS_10K if form == "10-K" else _ITEMS_10Q
    risks = risks if risks is not None else risk_paragraphs(rng, 18)
    parts: list[str] = [
        "<?xml version='1.0' encoding='ASCII'?>\n"
        '<html xmlns="http://www.w3.org/1999/xhtml"><head>'
        f"<title>{company} {form}</title></head><body>\n",
        '<div style="display:none"><ix:header><ix:hidden>'
        + "".join(
            f"<ix:nonNumeric>fact {rng.randint(0, 10**6)}</ix:nonNumeric>"
            for _ in range(30)
        )
        + "</ix:hidden></ix:header></div>\n",
        _p("UNITED STATES SECURITIES AND EXCHANGE COMMISSION", weight=700),
        _p(f"FORM {form}", weight=700),
        _p(f"{company} (Exact name of Registrant as specified in its charter)"),
    ]

    # Table of contents
    if variant != "no_toc":
        parts.append(_p("TABLE OF CONTENTS", weight=700))
        if variant == "inline_toc":
            for anchor, label, title in items:
                parts.append(f'<p><a href="#{anchor}">{label} {title}</a></p>\n')
        else:
            parts.append("<table>")
            for anchor, label, title in items:
                parts.append(
                    f'<tr><td><div><span><a href="#{anchor}">{label}</a></span></div></td>'
                    f'<td><div><span><a href="#{anchor}">{title}</a></span></div></td>'
                    f"<td><div><span>{rng.randint(1, 120)}</span></div></td></tr>"
                )
            parts.append("</table>\n")

    for anchor, label, title in items:
        heading_label, heading_title = label, title
        if variant == "caps_headings":
            heading_label, heading_title = label.upper(), title.upper()
        parts.append(f'<div id="{anchor}"></div>')
        if variant == "split_heading":
            parts.append(
                '<div style="margin-top:18pt">'
                + _SPAN.format(w=700, t=heading_label)
                + _SPAN.format(w=700, t="&#160;" + heading_title)
                + "</div>\n"
            )
        else:
            parts.append(
                '<div style="margin-top:18pt">'
                + _SPAN.format(
                    w=700, t=f"{heading_label}&#160;&#160;&#160;&#160;{heading_title}"
                )
                + "</div>\n"
            )
        if title == "Risk Factors":
            for i, para in enumerate(risks):
                if i % 4 == 0:
                    parts.append(
                        _p(
                            f"Risks Related to {rng.choice(_TOPICS).title()}",
                            weight=700,
                        )
                    )
                parts.append(_p(para))
        elif title.startswith("Financial Statements"):
            for _ in range(tables):
                parts.append(_table(rng, table_rows))
                parts.append(_p(_para(rng, 3)))
        elif title.startswith("Management"):
            for _ in range(body_paragraphs):
                parts.append(_p(_para(rng, rng.randint(3, 7))))
            parts.append(_table(rng, table_rows // 2))
        elif title in ("Business", "Legal Proceedings", "Controls and Procedures"):
            for _ in range(max(2, body_paragraphs // 2)):
                parts.append(_p(_para(rng, rng.randint(3, 6))))
        else:
            parts.append(_p(_para(rng, 2)))
    parts.append(_p("SIGNATURES", weight=700))
    parts.append("</body></html>\n")
    return SynthFiling(
        html="".join(parts),
        form=form,
        variant=variant,
        gold=dict(GOLD_10K if form == "10-K" else GOLD_10Q),
        risk_paragraphs=risks,
    )


def synth_corpus(
    n: int, *, seed: int = 0, variants: tuple[str, ...] = VARIANTS
) -> list[SynthFiling]:
    """``n`` synthetic filings cycling through forms and heading variants."""
    out = []
    for i in range(n):
        form = "10-K" if i % 3 != 2 else "10-Q"
        out.append(
            synth_filing(seed + i, form=form, variant=variants[i % len(variants)])
        )
    return out
