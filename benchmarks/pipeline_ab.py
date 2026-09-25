"""Repeated, interleaved A/B of the end-to-end pipeline's cold default run.

Runs the ``cold_default`` scenario of :mod:`benchmarks.bench_pipeline` in a
fresh subprocess per repetition, alternating between the installed source
("current") and another revision's ``src`` directory given with
``--baseline-src`` (exported with ``git archive <rev> src | tar -x -C DIR``).
Interleaving spreads machine-load drift evenly over both sides.

    python -m benchmarks.pipeline_ab --baseline-src /tmp/base/src --repeat 3
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmarks._common import REPO_ROOT, environment

_CHILD = """
import json
from benchmarks._common import quiet_logging
quiet_logging()
from benchmarks import bench_pipeline
import alphasig, os
r = bench_pipeline.run_scenario_cold_default()
from pathlib import Path
from benchmarks._common import _display_path
r["alphasig_path"] = _display_path(Path(alphasig.__file__).resolve().parent)
print(json.dumps(r))
"""


def _one(pythonpath: str | None) -> dict[str, Any]:
    env = dict(os.environ)
    paths = [str(REPO_ROOT)] + ([pythonpath] if pythonpath else [])
    env["PYTHONPATH"] = os.pathsep.join(paths[::-1])  # baseline src first
    out = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", _CHILD],
        env=env,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    result: dict[str, Any] = json.loads(out.stdout.strip().splitlines()[-1])
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline-src", required=True)
    ap.add_argument("--baseline-rev", default="baseline")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument(
        "--out", default=str(Path(__file__).parent / "results" / "pipeline_ab.json")
    )
    args = ap.parse_args(argv)

    runs: dict[str, list[dict[str, Any]]] = {"baseline": [], "current": []}
    for i in range(args.repeat):
        for side, path in (("baseline", args.baseline_src), ("current", None)):
            r = _one(path)
            runs[side].append(r)
            print(
                f"[ab] rep {i + 1} {side}: {r['wall_s']} s, {r['llm_calls']} LLM calls",
                file=sys.stderr,
            )

    def summary(rs: list[dict[str, Any]]) -> dict[str, Any]:
        walls = [r["wall_s"] for r in rs]
        return {
            "wall_s_median": statistics.median(walls),
            "wall_s_all": walls,
            "llm_calls": rs[0]["llm_calls"],
            "llm_max_inflight": max(r["llm_max_inflight"] for r in rs),
            "edgar_requests": rs[0]["edgar_requests"],
            "signals": rs[0]["signals"],
            "alphasig_path": rs[0]["alphasig_path"],
        }

    result = {
        "environment": environment(),
        "baseline_rev": args.baseline_rev,
        "repeat": args.repeat,
        "baseline": summary(runs["baseline"]),
        "current": summary(runs["current"]),
    }
    result["speedup_median"] = round(
        result["baseline"]["wall_s_median"] / result["current"]["wall_s_median"], 3
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: result[k] for k in ("baseline", "current", "speedup_median")}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
