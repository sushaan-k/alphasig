"""Shared helpers: environment capture, timing, quiet logging."""

from __future__ import annotations

import gc
import importlib.metadata
import logging
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED = 1234

_PACKAGES = (
    "alphasig",
    "beautifulsoup4",
    "lxml",
    "duckdb",
    "pyarrow",
    "networkx",
    "httpx",
    "pydantic",
    "anthropic",
    "respx",
    "pandas",
)


def quiet_logging() -> None:
    """Silence structlog/std logging so it does not skew timings."""
    import structlog

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL),
        cache_logger_on_first_use=False,
    )
    logging.disable(logging.CRITICAL)


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _display_path(path: Path) -> str:
    """Repo-relative path, or ``external:<last 3 parts>`` (no home/tmp dirs)."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return "external:" + "/".join(path.parts[-3:])


def environment() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for pkg in _PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    try:
        import alphasig

        alphasig_path = _display_path(Path(alphasig.__file__).resolve().parent)
    except ImportError:
        alphasig_path = None
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        # Set when benchmarking another revision's source via PYTHONPATH
        # (e.g. the pre-optimisation baseline); see docs/benchmarks.md.
        "code_under_test": os.environ.get("ALPHASIG_BENCH_CODE_REV")
        or _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "alphasig_import_path": alphasig_path,
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "packages": versions,
        "seed": SEED,
    }


def timeit(
    fn: Callable[[], Any], *, repeat: int = 5, warmup: int = 1
) -> dict[str, float]:
    """Run *fn* ``warmup + repeat`` times; return wall-clock stats in seconds."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeat):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return {
        "median_s": statistics.median(samples),
        "min_s": min(samples),
        "max_s": max(samples),
        "repeat": repeat,
    }
