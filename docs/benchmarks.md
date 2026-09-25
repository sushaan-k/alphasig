# Benchmarks

alphasig ships an offline, reproducible benchmark suite under
[`benchmarks/`](../benchmarks). It needs no access to sec.gov and no LLM API
key: EDGAR is mocked with `respx` and the Anthropic client is replaced by a
fake with injected latency. Every number on this page comes from a committed
result file in [`benchmarks/results/`](../benchmarks/results); nothing is
estimated or extrapolated.

| file | what it measures |
|---|---|
| [`after.json`](../benchmarks/results/after.json) / [`.md`](../benchmarks/results/after.md) | The full suite on the current 0.2.0 code (commit `4f86ed0`, clean tree). |
| [`baseline.json`](../benchmarks/results/baseline.json) / [`.md`](../benchmarks/results/baseline.md) | **The 0.1.x baseline:** the same suite against the source at `54efd36` (the 0.1.x code base just renamed to `alphasig`, before the 0.2.0 fixes and optimisations), run with an earlier revision of this harness. |
| [`pipeline_ab.json`](../benchmarks/results/pipeline_ab.json) | The end-to-end pipeline, `54efd36` and current code interleaved, 3 runs each. |
| [`parser-b928903.json`](../benchmarks/results/parser-b928903.json) / [`.md`](../benchmarks/results/parser-b928903.md) | Parser and boundary benches on `b928903`: the 0.2.0 BeautifulSoup parser just before the single-pass lxml scan. |

## Hardware and noise

All results were produced in **a shared 4-vCPU cloud container** (Intel
Xeon @ 2.80 GHz, 16 GB RAM, Linux, CPython 3.11.15), not a dedicated
benchmark machine. Other workloads may share the host. Timings are medians,
usually of 3 to 7 repetitions (see `repeat` in the JSON), except that the
storage insert and every pipeline scenario in `after.json` are single runs.
Unchanged code gives a noise estimate: the graph bench did not change
between the baseline and current runs, yet `exposure()` on the 100k-edge
graph measured 90 ms and then 82 ms, and `from signals` 1.68 s and then
1.09 s. Treat differences under about 1.5× as noise unless they are
repeated, as in `pipeline_ab.json`.

## Running

```bash
uv sync --extra dev          # or: pip install -e ".[dev]"

python -m benchmarks.run                    # full suite (~5 min on 4 vCPUs)
python -m benchmarks.run --quick            # smoke run (~35 s), used in CI
python -m benchmarks.run --only parser,storage --label mylabel
python -m benchmarks.render benchmarks/results/mylabel.json   # JSON -> Markdown
```

Each run writes `benchmarks/results/<label>.json` and `<label>.md`. The JSON
records the UTC timestamp, git commit, whether the tracked tree was dirty,
the source revision under test and its import path, the Python version, the
platform, the CPU model and count, the versions of every relevant package,
and the RNG seed (`1234`).

To benchmark another revision, export its `src/` and put it first on
`PYTHONPATH`; the harness itself stays at the current revision:

```bash
git archive 54efd36 src | tar -x -C /tmp/base
PYTHONPATH=/tmp/base/src ALPHASIG_BENCH_CODE_REV=54efd36 \
    python -m benchmarks.run --label baseline
python -m benchmarks.pipeline_ab --baseline-src /tmp/base/src \
    --baseline-rev 54efd36 --repeat 3
```

## Fixtures: what is real and what is synthetic

| bench | data |
|---|---|
| parser throughput | **real** (5 filings) and **synthetic** (40 filings), reported separately |
| Item 1A / Item 7 boundaries | **real** (5 filings) and **synthetic** (50 filings), reported separately |
| risk-diff throughput | **synthetic** risk-factor text only |
| DuckDB store | **synthetic** signals (3,000 tickers, 4 types, ~15 years of timestamps) |
| supply-chain graph | **synthetic** edges (Pareto-distributed suppliers, ~10% duplicates) |
| end-to-end pipeline | **synthetic** filings, mocked EDGAR, fake LLM |

- *Real:* five public filings (Apple 10-K and 10-Q, NVIDIA 10-K, Oracle 10-K
  and 10-Q; 13.1 MB of inline-XBRL HTML) that the MIT-licensed
  [`dgunning/edgartools`](https://github.com/dgunning/edgartools) project
  ships as test data. They are fetched from `raw.githubusercontent.com` at the
  pinned tag `v5.58.0`, verified against SHA-256 checksums in
  [`benchmarks/fixtures.py`](../benchmarks/fixtures.py), and cached under the
  git-ignored `benchmarks/.cache/`. A file whose checksum does not match is
  never used. If the download fails, the benches continue with synthetic data
  only and record `real_fixtures_available: 0`.
- *Synthetic:* generated deterministically from the seed, mimicking
  Workiva-style markup: nested `<div><span style=…>`, a table of contents,
  financial tables, hidden `ix:header` facts and anchor ids on every Item
  heading. There are five heading/TOC variants: `workiva`, `inline_toc`,
  `caps_headings`, `split_heading` and `no_toc`. Between a company's
  filings, about 10% of risk paragraphs escalate "may" to "is currently", one
  paragraph is dropped and one is added. Synthetic text is not real
  disclosure language: treat those results as engineering measurements, not
  claims about extraction quality on real filings.

**Boundary ground truth.** A gold span runs from the element that the
filing's own table-of-contents link targets for Item 1A or Item 7 (Part I
Item 2 for 10-Q MD&A) to the element targeted by the next Item's link. The
ids for the real filings were chosen by inspecting each document and are
stored in `REAL_FIXTURES`. Gold spans and parser output are compared as
character offsets in the same flattened text. (The `-1` end errors in the
results are an artefact of that flattening: the gold end offset includes the
separator before the next heading.)

**Mocks.**

- *EDGAR:* `company_tickers.json`, `submissions/CIK*.json` and Archives
  documents are served with 20 ms latency per request. The real
  `EdgarClient`, including its 10 requests/second limiter, stays in the loop.
- *LLM:* `anthropic.AsyncAnthropic` is replaced by a fake that sleeps a fixed
  250 ms per call (50 ms in `--quick`) and returns schema-valid JSON for each
  engine's prompt. It counts calls and peak in-flight requests. Prompt
  caching, retries and token costs are not modelled.

## Results: 0.1.x baseline vs. 0.2.0

### End-to-end pipeline

6 tickers × 6 filings (3 10-K and 3 10-Q each), all four engines, mock LLM
latency 250 ms per call. From
[`pipeline_ab.json`](../benchmarks/results/pipeline_ab.json), the two versions
interleaved, 3 runs each:

| | 0.1.x baseline `54efd36` | 0.2.0 | change |
|---|---|---|---|
| wall time, median (s) | 26.70 (26.81, 25.71, 26.70) | 13.78 (13.77, 13.81, 13.78) | **1.94× faster** |
| LLM API calls | 372 | 366 | −1.6% |
| peak LLM requests in flight | 12 (uncapped) | 8 (`llm_concurrency`) | bounded |
| EDGAR requests | 45 | 43 | ticker map downloaded once |
| signals | 120 | 120 | |

Scenarios on the current code (single runs, from `after.json`):

| scenario | wall s | LLM calls | EDGAR requests |
|---|---|---|---|
| cold EDGAR cache, defaults (`llm_concurrency=8`) | 13.7 | 366 | 43 |
| warm EDGAR disk cache | 12.3 | 366 | 7 (no document downloads) |
| `llm_concurrency=1` | 93.9 | 366 | 43 |
| `llm_concurrency=4` | 25.2 | 366 | 43 |
| `llm_concurrency=16` | 8.3 | 366 | 43 |
| re-run with `llm_cache_dir` | 3.0 | **0** | 7 |
| `incremental=True` re-run | **0.71** | 0 | 7 |

With `llm_concurrency=1` the run is LLM-bound (366 × 0.25 s = 91.5 s of
model latency against 93.9 s of wall time). At 16 it takes 8.3 s against
5.7 s of latency per slot, so the remaining time is parsing, the risk
differ's similarity check and the EDGAR rate limit. An LLM-cached re-run
still downloads submissions and re-parses filings (3.0 s); an incremental
re-run sees that every (filing, engine) job is recorded and downloads no
documents at all.

### Parser / section extraction

The 0.2.0 parser fixes real bugs, so its output differs from 0.1.x (see the
boundary results). Its section scan was then rewritten to walk the document
once with lxml instead of the BeautifulSoup tree once per candidate heading;
`tests/test_parser_equivalence.py` requires that rewrite's output to be
identical to the BeautifulSoup implementation on all five real filings, the
synthetic variants and a messy-markup case.

| input | 0.1.x `54efd36` | 0.2.0 before rewrite `b928903` | 0.2.0 | vs. `b928903` |
|---|---|---|---|---|
| Apple 10-K (1.9 MB) | 0.408 s | 0.444 s | 0.084 s | 5.3× |
| Apple 10-Q (1.1 MB) | 0.288 s | 0.256 s | 0.041 s | 6.2× |
| NVIDIA 10-K (2.7 MB) | 0.474 s | 0.476 s | 0.094 s | 5.1× |
| Oracle 10-K (5.3 MB) | 1.328 s | 1.367 s | 0.214 s | 6.4× |
| Oracle 10-Q (2.0 MB) | 0.427 s | 0.462 s | 0.080 s | 5.8× |
| all 5 real (13.1 MB) | 3.66 s (3.6 MB/s) | 3.57 s (3.7 MB/s) | **0.47 s (27.7 MB/s)** | 7.6× |
| 40 synthetic (25.5 MB) | 9.93 s (2.6 MB/s) | 9.69 s (2.6 MB/s) | **1.66 s (15.3 MB/s)** | 5.8× |

8-K filings are still parsed with BeautifulSoup (whole document as one
section) and are not covered by these numbers.

### DuckDB `SignalStore`

| signals | insert, 0.1.x (`executemany`) | insert, 0.2.0 (Arrow, idempotent) | re-insert same batch (all skipped) | 30-day range query | `query()` → `Signal` objects | `to_arrow()` |
|---|---|---|---|---|---|---|
| 10,000 | 57.2 s (175 rows/s) | 0.55 s (18,178 rows/s) | 0.08 s | 9.2 → 2.6 ms | 0.19 → 0.21 s | 13.7 ms |
| 100,000 | not run (too slow) | 1.42 s (70,224 rows/s) | 0.88 s | 33.9 ms | 2.04 s | 103 ms |
| 1,000,000 | not run (too slow) | 12.0 s (83,083 rows/s) | 9.0 s | 59.7 ms | 19.2 s | 945 ms |

Insert times are single runs. The 0.1.x row-by-row path was not run at 100k
or 1M rows. `to_arrow()` returns every matching row 15-20× faster than
building `Signal` objects with `query()`, and is the path to use for
analytics (`to_pandas()` wraps it). Latencies for every query shape at every
size are in `after.md`.

### Risk-diff throughput

The similarity check that decides whether a risk-factor pair reaches the LLM
is `difflib.SequenceMatcher` over words in 0.2.0 (over characters in 0.1.x).
It now runs in a worker thread so it no longer blocks other filings' I/O, and
identical text returns immediately.

| pair (synthetic) | chars (both) | 0.1.x s | 0.2.0 s | 0.2.0 similarity |
|---|---|---|---|---|
| evolved, 10 paragraphs | 30k | 0.041 | 0.021 | 0.924 |
| evolved, 40 paragraphs | 120k | 2.47 | 0.28 | 0.980 |
| evolved, 80 paragraphs | 235k | 14.1 | 1.14 | 0.988 |
| identical, 40 paragraphs | 120k | 0.216 | **0.0011** | 1.0 |
| engine, 7 consecutive 10-K pairs (~68k chars each), zero-latency LLM | | 19.2 s, 7 LLM calls | 2.8 s, 7 LLM calls | |

On a real section (Apple's 10-K Item 1A, 69k characters) the check takes
about 0.15 s, and 1 ms when the text is unchanged.

### Item 1A / Item 7 boundaries

| corpus | section | 0.1.x detected | 0.1.x precision / recall | 0.2.0 detected | 0.2.0 precision / recall |
|---|---|---|---|---|---|
| real | Item 1A risk factors | 4 / 5 | 0.748 / 0.800 | 5 / 5 | 1.000 / 0.999 |
| real | Item 7 / Part I Item 2 MD&A | 5 / 5 | 0.798 / 1.000 | 5 / 5 | 1.000 / 1.000 |
| synthetic | risk factors | 43 / 50 | 0.917 / 0.860 | 50 / 50 | 1.000 / 1.000 |
| synthetic | MD&A | 50 / 50 | 0.773 / 0.800 | 50 / 50 | 1.000 / 1.000 |

Every section 0.2.0 finds is exact to within one character (the `-1`
flattening artefact above), except NVIDIA's MD&A, which starts 18 characters
late. The 0.2.0 columns and the engine row above are from
`results/gate-and-heading-fix.md`, measured after the two fixes described
under Findings.

### Supply-chain graph

This code did not change. 100k input edges (63.5k kept after confidence
merging, 10.3k nodes) build in 0.52 s; `exposure()` on the biggest hub takes
82 ms and `most_connected(10)` 33 ms.

## Findings

The benchmarks exposed two extraction bugs, both now fixed:

1. **Oracle's Item 1A was not found** (10-K and 10-Q). Oracle styles the
   first letter(s) of each heading word separately, so the heading text read
   `Item 1A. R isk Factors` / `Ri sk Factors` and did not match. Heading
   matching now rejoins such fragments; all 5 real filings' Item 1A are found
   (previously 3 of 5).
2. **The risk-diff gate skipped material changes.** The LLM was only called
   when word similarity was at most 0.98, a ratio that shrinks with section
   length: a new ~570-character risk paragraph in Apple's 69k-character
   Item 1A scored 0.995 and never reached the model, and 5 of 7 synthetic
   consecutive-year pairs were skipped. The gate now counts changed words,
   ignoring purely numeric tokens (years, amounts): the LLM is skipped only
   when fewer than 5 words of wording changed, so date and figure updates
   still cost nothing and all 7 synthetic pairs are analysed.

Still open:

1. The similarity check is still super-linear: 1.1 s at 235k characters and
   about 11 s at 600k. It no longer blocks the event loop, but it holds the
   GIL while it runs.

## Point-in-time alignment

Signals are stamped with `FilingSection.available_at`: the EDGAR acceptance
time, or 17:30 ET on the filing date when the acceptance time is missing
(EDGAR dates filings accepted after 17:30 to the next business day, so the
fallback never precedes publication). The event-study hook below aligns each
signal to the first 16:00 ET close strictly after its timestamp;
`tests/test_event_study_alignment.py` covers the before/after-close cases.

## Real-data harness (scaffold only, no results)

[`benchmarks/real_data/`](../benchmarks/real_data) holds two scripts that need
live data. They have not been run and **no numbers are reported for them**.
Both refuse to run without their inputs.

- `cost_quality.py` needs `ALPHASIG_USER_AGENT` and `ANTHROPIC_API_KEY`. Per
  filing it records document size, parse time, sections found, API calls,
  response-cache hits, input/output/prompt-cache-read tokens as reported by
  the API, engine wall time and signals. Dollar cost is computed only from
  the `--price-in-per-mtok` / `--price-out-per-mtok` you pass, and
  precision/recall only against a JSONL label file you supply.
- `event_study.py` takes `--prices` (a daily price CSV you supply, with
  columns `date,ticker,close`) and optionally `--market-ticker`, and reports
  mean market-adjusted cumulative abnormal returns, signed by signal
  direction, with t-statistics per signal type, direction and window. It
  reads signals from Parquet or from a DuckDB store via
  `SignalStore.to_arrow()`.
