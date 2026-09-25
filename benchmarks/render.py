"""Render benchmark JSON as Markdown tables.

Usage::

    python -m benchmarks.render benchmarks/results/<label>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    def fmt(v: Any) -> str:
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:,.4g}" if abs(v) < 1000 else f"{v:,.0f}"
        if isinstance(v, int) and not isinstance(v, bool):
            return f"{v:,}"
        return str(v)

    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def _env(env: dict[str, Any]) -> str:
    pk = env.get("packages", {})
    rows: list[list[Any]] = [
        ["timestamp (UTC)", env.get("timestamp_utc")],
        [
            "git commit",
            f"{env.get('git_commit', '')[:12]}{' (dirty)' if env.get('git_dirty') else ''}",
        ],
        ["code under test", str(env.get("code_under_test", ""))[:12]],
        ["alphasig imported from", env.get("alphasig_import_path")],
        ["python", f"{env.get('python')} ({env.get('implementation')})"],
        ["platform", env.get("platform")],
        ["cpu", f"{env.get('cpu_model')} x{env.get('cpu_count')}"],
        ["seed", str(env.get("seed"))],
        ["packages", ", ".join(f"{k} {v}" for k, v in pk.items() if v)],
    ]
    return _table(["field", "value"], rows)


def _parser(r: dict[str, Any]) -> str:
    rows = []
    for name, d in (r.get("real_per_document") or {}).items():
        rows.append(
            [
                f"real: {name}",
                d["total_mb"],
                d["median_s"],
                d["mb_per_s"],
                d["filings_per_s"],
            ]
        )
    for key, label in (
        ("real_corpus", "real: all 5"),
        ("synthetic_corpus", "synthetic corpus"),
    ):
        if key in r:
            d = r[key]
            rows.append(
                [
                    f"{label} ({d['filings']} filings)",
                    d["total_mb"],
                    d["median_s"],
                    d["mb_per_s"],
                    d["filings_per_s"],
                ]
            )
    return _table(["input", "MB", "median s", "MB/s", "filings/s"], rows)


def _boundaries(r: dict[str, Any]) -> str:
    out = []
    rows = []
    for corpus in ("real_summary", "synthetic_summary"):
        for sec, d in (r.get(corpus) or {}).items():
            rows.append(
                [
                    corpus.split("_")[0],
                    sec,
                    d["documents"],
                    d["detected"],
                    d["mean_precision"],
                    d["mean_recall"],
                    d["near_exact_(p,r>=0.99)"],
                ]
            )
    out.append(
        _table(
            [
                "corpus",
                "section",
                "docs",
                "detected",
                "mean precision",
                "mean recall",
                "near-exact",
            ],
            rows,
        )
    )
    per_doc = r.get("real_per_document") or {}
    if per_doc:
        rows = []
        for doc, secs in per_doc.items():
            for sec, d in secs.items():
                rows.append(
                    [
                        doc,
                        sec,
                        d.get("precision"),
                        d.get("recall"),
                        d.get("start_error_chars"),
                        d.get("end_error_chars"),
                        d.get("gold_chars"),
                    ]
                )
        out.append("\nPer real document (errors in characters; + = past gold end):\n\n")
        out.append(
            _table(
                [
                    "document",
                    "section",
                    "precision",
                    "recall",
                    "start err",
                    "end err",
                    "gold chars",
                ],
                rows,
            )
        )
    by_var = r.get("synthetic_by_variant") or {}
    if by_var:
        rows = []
        for var, secs in by_var.items():
            for sec, d in secs.items():
                rows.append(
                    [
                        var,
                        sec,
                        d["documents"],
                        d["detected"],
                        d["mean_precision"],
                        d["mean_recall"],
                    ]
                )
        out.append("\nSynthetic, by heading/TOC variant:\n\n")
        out.append(
            _table(
                [
                    "variant",
                    "section",
                    "docs",
                    "detected",
                    "mean precision",
                    "mean recall",
                ],
                rows,
            )
        )
    return "".join(out)


def _risk(r: dict[str, Any]) -> str:
    rows = [
        [
            d.get("pair", "evolved"),
            d["paragraphs"],
            d["chars_prev"] + d["chars_cur"],
            d["similarity"],
            d["median_s"],
            d["chars_per_s"],
        ]
        for d in r.get("similarity_gate", [])
    ]
    out = _table(
        ["pair", "paragraphs", "chars (both)", "similarity", "median s", "chars/s"],
        rows,
    )
    e = r.get("engine")
    if e:
        out += "\n" + _table(
            [
                "filing pairs",
                "avg section chars",
                "LLM calls",
                "signals",
                "median s",
                "pairs/s",
            ],
            [
                [
                    e["filing_pairs"],
                    e["avg_section_chars"],
                    e["llm_calls"],
                    e["signals"],
                    e["median_s"],
                    e["pairs_per_s"],
                ]
            ],
        )
    return out


def _storage(r: dict[str, Any]) -> str:
    rows = []
    for d in r.get("sizes", []):
        if "skipped" in d:
            rows.append([d["signals"], "skipped: " + d["skipped"]] + [None] * 9)
            continue
        rows.append(
            [
                d["signals"],
                d["insert_s"],
                d["insert_rows_per_s"],
                d.get("reinsert_same_s"),
                d["ticker_eq_ms"],
                d["type_and_min_strength_ms"],
                d["time_range_30d_ms"],
                d["latest_1000_ms"],
                d["summary_ms"],
                d["query_all_to_signals_s"],
                d.get("to_arrow_all_ms"),
            ]
        )
    head = f"Insert path: `{r.get('insert_path')}`\n\n"
    return head + _table(
        [
            "signals",
            "insert s",
            "rows/s",
            "re-insert s",
            "ticker= ms",
            "type+strength ms",
            "30d range ms",
            "latest 1k ms",
            "summary ms",
            "all→Signal s",
            "all→Arrow ms",
        ],
        rows,
    )


def _graph(r: dict[str, Any]) -> str:
    rows = [
        [
            d["edges_in"],
            d["nodes"],
            d["edges_kept"],
            d["build_s"],
            d["edges_per_s"],
            d["exposure_hub_ms"],
            d["most_connected_ms"],
            d["from_signal_collection_s"],
        ]
        for d in r.get("sizes", [])
    ]
    return _table(
        [
            "edges in",
            "nodes",
            "edges kept",
            "build s",
            "edges/s",
            "exposure(hub) ms",
            "top-10 ms",
            "from signals s",
        ],
        rows,
    )


def _pipeline(r: dict[str, Any]) -> str:
    head = (
        f"{r.get('tickers')} tickers x {r.get('filings_per_ticker')} filings, "
        f"mock LLM latency {r.get('llm_latency_s')} s/call, mock network latency "
        f"{r.get('net_latency_s')} s/request, EDGAR limiter active.\n\n"
    )
    rows = []
    for name, d in r.get("scenarios", {}).items():
        if "skipped" in d:
            rows.append([name, "skipped: " + d["skipped"]] + [None] * 6)
            continue
        rows.append(
            [
                name,
                d["wall_s"],
                d["llm_calls"],
                d["llm_max_inflight"],
                d["edgar_requests"],
                d["edgar_archive_requests"],
                d["signals"],
            ]
        )
    return head + _table(
        [
            "scenario",
            "wall s",
            "LLM calls",
            "peak LLM in-flight",
            "EDGAR requests",
            "archive fetches",
            "signals",
        ],
        rows,
    )


_RENDERERS = {
    "parser": ("Parser / section extraction throughput", _parser),
    "boundaries": ("Item 1A / Item 7 boundary precision & recall", _boundaries),
    "risk_diff": ("Risk-diff throughput", _risk),
    "storage": ("DuckDB SignalStore latency", _storage),
    "graph": ("Supply-chain graph construction", _graph),
    "pipeline": (
        "End-to-end pipeline (mock EDGAR + latency-injected mock LLM)",
        _pipeline,
    ),
}


def render_markdown(results: dict[str, Any]) -> str:
    parts = [
        "# alphasig offline benchmark results\n\n",
        *([f"{results['note']}\n\n"] if results.get("note") else []),
        f"Mode: {'quick' if results.get('quick') else 'full'}\n\n",
        "## Environment\n\n",
        _env(results.get("environment", {})),
    ]
    for name, res in results.get("benches", {}).items():
        title, fn = _RENDERERS.get(name, (name, None))
        parts.append(f"\n## {title}\n\n")
        if "error" in res:
            parts.append("```\n" + res["error"] + "\n```\n")
        elif fn:
            parts.append(fn(res))
        parts.append(f"\n_bench wall time: {res.get('bench_wall_s')} s_\n")
    return "".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    args = ap.parse_args()
    print(render_markdown(json.loads(Path(args.result).read_text())))


if __name__ == "__main__":
    main()
