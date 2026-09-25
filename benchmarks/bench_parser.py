"""Parser / section-extraction throughput (MB/s and filings/s)."""

from __future__ import annotations

from datetime import date
from typing import Any

from alphasig.models import Filing, FilingType
from alphasig.parser import parse_filing
from benchmarks._common import SEED, timeit
from benchmarks.fixtures import load_real_fixtures, synth_corpus


def make_filing(html: str, form: str, idx: int = 0) -> Filing:
    return Filing(
        accession_number=f"0000000000-24-{idx:06d}",
        cik="0000000000",
        ticker="BENCH",
        company_name="Bench Corp",
        filing_type=FilingType(form),
        filed_date=date(2024, 1, 1),
        period_of_report=date(2023, 12, 31),
        url=f"https://example.invalid/{idx}.htm",
        raw_html=html,
    )


def _throughput(filings: list[Filing], repeat: int) -> dict[str, Any]:
    total_bytes = sum(len(f.raw_html.encode("utf-8")) for f in filings)
    stats = timeit(lambda: [parse_filing(f) for f in filings], repeat=repeat)
    t = stats["median_s"]
    return {
        "filings": len(filings),
        "total_mb": round(total_bytes / 1e6, 3),
        "median_s": round(t, 4),
        "min_s": round(stats["min_s"], 4),
        "mb_per_s": round(total_bytes / 1e6 / t, 3),
        "filings_per_s": round(len(filings) / t, 3),
        "repeat": repeat,
    }


def run(quick: bool = False) -> dict[str, Any]:
    repeat = 2 if quick else 5
    out: dict[str, Any] = {"repeat": repeat}

    real = load_real_fixtures()
    out["real_fixtures_available"] = len(real)
    if real:
        per_doc = {}
        for fx, html in real:
            per_doc[fx.name] = _throughput([make_filing(html, fx.form)], repeat)
        out["real_per_document"] = per_doc
        out["real_corpus"] = _throughput(
            [make_filing(html, fx.form, i) for i, (fx, html) in enumerate(real)],
            repeat,
        )

    n = 10 if quick else 40
    corpus = synth_corpus(n, seed=SEED)
    out["synthetic_corpus"] = _throughput(
        [make_filing(s.html, s.form, i) for i, s in enumerate(corpus)], repeat
    )
    return out
