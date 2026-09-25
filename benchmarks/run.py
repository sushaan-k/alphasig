"""Run the offline benchmark suite and write JSON + Markdown results.

Usage::

    python -m benchmarks.run                 # full suite
    python -m benchmarks.run --quick         # small sizes, for CI smoke runs
    python -m benchmarks.run --only parser,storage --label after

Each run writes ``<out>/<label>.json`` (raw results plus environment:
package versions, CPU model, git commit, UTC timestamp, seed) and
``<out>/<label>.md`` (tables rendered from that JSON by
:mod:`benchmarks.render`).  Everything runs offline except the optional,
checksum-verified download of the pinned real fixtures.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from benchmarks import (
    bench_boundaries,
    bench_graph,
    bench_parser,
    bench_pipeline,
    bench_risk_diff,
    bench_storage,
)
from benchmarks._common import environment, quiet_logging
from benchmarks.render import render_markdown

BENCHES: dict[str, Callable[[bool], dict[str, Any]]] = {
    "parser": bench_parser.run,
    "boundaries": bench_boundaries.run,
    "risk_diff": bench_risk_diff.run,
    "storage": bench_storage.run,
    "graph": bench_graph.run,
    "pipeline": bench_pipeline.run,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="small sizes (CI smoke run)")
    ap.add_argument(
        "--only", default="", help="comma-separated subset of: " + ",".join(BENCHES)
    )
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    ap.add_argument(
        "--label", default=None, help="output file stem (default: timestamp)"
    )
    ap.add_argument("--note", default=None, help="free-text note stored in the JSON")
    args = ap.parse_args(argv)

    quiet_logging()
    selected = [b for b in args.only.split(",") if b] or list(BENCHES)
    unknown = set(selected) - set(BENCHES)
    if unknown:
        ap.error(f"unknown bench(es): {sorted(unknown)}")

    env = environment()
    results: dict[str, Any] = {"environment": env, "quick": args.quick, "benches": {}}
    if args.note:
        results["note"] = args.note
    failed = False
    for name in selected:
        print(f"[bench] {name} ...", file=sys.stderr, flush=True)
        t0 = time.perf_counter()
        try:
            res = BENCHES[name](args.quick)
        except Exception:  # keep the other benches running
            failed = True
            res = {"error": traceback.format_exc(limit=5)}
        res["bench_wall_s"] = round(time.perf_counter() - t0, 2)
        results["benches"][name] = res
        print(
            f"[bench] {name} done in {res['bench_wall_s']}s",
            file=sys.stderr,
            flush=True,
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or env["timestamp_utc"].replace(":", "").replace("+0000", "Z")
    json_path = out_dir / f"{label}.json"
    json_path.write_text(json.dumps(results, indent=2, sort_keys=False) + "\n")
    md_path = out_dir / f"{label}.md"
    md_path.write_text(render_markdown(results))
    print(f"[bench] wrote {json_path} and {md_path}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
