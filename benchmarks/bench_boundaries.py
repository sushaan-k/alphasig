"""Item 1A (Risk Factors) / Item 7 (MD&A) boundary precision and recall.

Ground truth is an anchor-id span: the section runs from the element with
the start id (the target of the table-of-contents link for that Item) to the
element with the next Item's id.  Both the gold span and the parser output
are expressed as character offsets into the same flattened document text
(visible text chunks, stripped, joined by one space, whitespace runs
collapsed; the convention :mod:`alphasig.parser` uses), so the score is a
character-level overlap:

* precision = |pred ∩ gold| / |pred|
* recall    = |pred ∩ gold| / |gold|

A section the parser does not return counts as recall 0 and is excluded
from precision (reported separately as ``detected``).
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from lxml import etree

from alphasig.parser import find_section, parse_filing
from benchmarks._common import SEED
from benchmarks.bench_parser import make_filing
from benchmarks.fixtures import VARIANTS, load_real_fixtures, synth_corpus

_WS = re.compile(r"\s{2,}")
TARGET_SECTIONS = ("risk_factors", "md_and_a")


def flatten_with_anchors(html: str) -> tuple[str, dict[str, int]]:
    """Flatten *html* to text, returning offsets of every id/name anchor."""
    parser = etree.HTMLParser(recover=True)
    parser.feed(html)
    root = parser.close()
    pieces: list[str] = []
    offset = 0
    anchors: dict[str, int] = {}

    def add(raw: str | None) -> None:
        nonlocal offset
        if not raw:
            return
        chunk = raw.strip()
        if not chunk:
            return
        chunk = _WS.sub(" ", chunk)
        if pieces:
            offset += 1
        pieces.append(chunk)
        offset += len(chunk)

    for event, node in etree.iterwalk(root, events=("start", "end")):
        if event == "start":
            if isinstance(node.tag, str):
                for attr in ("id", "name"):
                    key = node.get(attr)
                    if key and key not in anchors:
                        # Offset of the first character at/after this element.
                        anchors[key] = offset + (1 if pieces else 0)
            add(node.text)
        else:
            add(node.tail)
    return " ".join(pieces), anchors


def score_document(
    html: str, form: str, gold: dict[str, tuple[str, str]]
) -> dict[str, Any]:
    flat, anchors = flatten_with_anchors(html)
    sections = parse_filing(make_filing(html, form))
    result: dict[str, Any] = {}
    for key in TARGET_SECTIONS:
        if key not in gold:
            continue
        start_id, end_id = gold[key]
        g0, g1 = anchors[start_id], anchors[end_id]
        sec = find_section(sections, key)
        if sec is None:
            result[key] = {"detected": False, "gold_chars": g1 - g0}
            continue
        pos = flat.find(sec.text[:400])
        if pos < 0:
            result[key] = {"detected": True, "locatable": False, "gold_chars": g1 - g0}
            continue
        p0, p1 = pos, pos + len(sec.text)
        overlap = max(0, min(p1, g1) - max(p0, g0))
        result[key] = {
            "detected": True,
            "locatable": True,
            "precision": overlap / (p1 - p0),
            "recall": overlap / (g1 - g0),
            "start_error_chars": p0 - g0,
            "end_error_chars": p1 - g1,
            "gold_chars": g1 - g0,
            "pred_chars": p1 - p0,
        }
    return result


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    det = [r for r in rows if r.get("detected") and r.get("locatable", True)]
    precision = sum(r["precision"] for r in det) / len(det) if det else None
    recall = sum(r.get("recall", 0.0) for r in rows) / n if n else None
    exact = sum(1 for r in det if r["precision"] >= 0.99 and r["recall"] >= 0.99)
    return {
        "documents": n,
        "detected": len(det),
        "mean_precision": None if precision is None else round(precision, 4),
        "mean_recall": None if recall is None else round(recall, 4),
        "near_exact_(p,r>=0.99)": exact,
    }


def run(quick: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {}

    real = load_real_fixtures()
    out["real_fixtures_available"] = len(real)
    if real:
        per_doc: dict[str, Any] = {}
        by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for fx, html in real:
            scored = score_document(html, fx.form, fx.gold)
            per_doc[fx.name] = {
                k: {
                    kk: (round(vv, 4) if isinstance(vv, float) else vv)
                    for kk, vv in v.items()
                }
                for k, v in scored.items()
            }
            for key, row in scored.items():
                by_section[key].append(row)
        out["real_per_document"] = per_doc
        out["real_summary"] = {k: _aggregate(v) for k, v in by_section.items()}

    n = 10 if quick else 50
    corpus = synth_corpus(n, seed=SEED)
    by_variant: dict[str, dict[str, list[dict[str, Any]]]] = {
        v: defaultdict(list) for v in VARIANTS
    }
    overall: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in corpus:
        for key, row in score_document(s.html, s.form, s.gold).items():
            by_variant[s.variant][key].append(row)
            overall[key].append(row)
    out["synthetic_documents"] = n
    out["synthetic_summary"] = {k: _aggregate(v) for k, v in overall.items()}
    out["synthetic_by_variant"] = {
        variant: {k: _aggregate(v) for k, v in secs.items()}
        for variant, secs in by_variant.items()
    }
    return out
