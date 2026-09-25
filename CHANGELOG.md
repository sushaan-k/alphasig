# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-09-25

### Changed

- **Renamed the import package and CLI from `sigint` to `alphasig`** so the
  distribution, import and command names match (`sigint` on PyPI belongs to
  another project). Migration:

  | 0.1.x | 0.2.0 |
  |---|---|
  | `from sigint import Pipeline` | `from alphasig import Pipeline` |
  | `import sigint.storage` | `import alphasig.storage` |
  | `sigint extract ...` | `alphasig extract ...` (`sigint` still works as an alias) |
  | default DB `sigint.duckdb` | `alphasig.duckdb` (rename your file or pass `--db`) |

- Signal `timestamp` is now the time the filing became public (EDGAR
  acceptance time, UTC) instead of midnight UTC on the filing date.
- A declared EDGAR User-Agent is required (`user_agent=`, `--user-agent` or
  `ALPHASIG_USER_AGENT`); the shared placeholder default was removed.
- Default model is `claude-sonnet-5` (override with `model=`, `--model` or
  `ALPHASIG_MODEL`). Requires `anthropic>=1`.
- Amendment dedupe keeps the earliest copy of a duplicated signal.
- `SignalStore.insert` skips signals that are already stored and returns the
  number of new rows.

### Added

- Optional Jev calibration (`alphasig[jev]`): `JevCalibrator`,
  `Pipeline(calibrator=...)` and `alphasig extract --calibrate
  [--drop-below P]` replace LLM self-reported confidence with TypeSafe Jev's
  calibrated probability that each signal's claim is supported by the filing.
- `alphasig extract --tickers AAPL MSFT` (as documented), comma-separated
  tickers and positional tickers.
- `half_life_days` for `rank_signals` / `summarize_sector_exposure` and
  `--half-life` for `alphasig rank` / `alphasig sectors`.
- `SupplyChainEdge.exposure`: the concentration share stated in the filing
  (e.g. 22% of net sales), carried into signal metadata and graph edges.
- `Filing.accepted_at`, `Filing.available_at` and `FilingSection.available_at`.
- 8-K filings are parsed as a single `current_report` section.
- `SignalStore` is a context manager; `LLMClient` tracks cached input tokens.
- `SignalStore.to_arrow()` / `to_pandas()`: query results as a pyarrow Table
  or pandas DataFrame (same schema as the Parquet export) without building
  `Signal` objects. pandas is the optional `alphasig[pandas]` extra.
- Incremental extraction: `Pipeline.extract(incremental=True)` /
  `alphasig extract --incremental` records finished (filing, engine) jobs in
  an `extraction_log` table with their signals and skips them next time, so
  re-runs resume after a crash or pick up only new filings.
- LLM response cache: `Pipeline(llm_cache_dir=...)` / `--llm-cache-dir`
  stores responses on disk keyed by a hash of the full request (model,
  prompts and parameters); re-runs reuse them instead of calling the API.
- `Pipeline(llm_concurrency=...)` / `--llm-concurrency` sets the maximum
  in-flight LLM requests (default 8).
- An offline benchmark suite (`python -m benchmarks.run`, see
  `docs/benchmarks.md`), run in CI in `--quick` mode.
- `py.typed`, `alphasig --version`, a wheel smoke-test CI job and a
  trusted-publishing release workflow.

### Fixed

- Section parser: a table of contents hid Risk Factors / MD&A, Items 1B/1C
  leaked into Risk Factors, and running page headers split sections.
- Section parser: headings that style a word's first letter(s) separately
  (`R isk Factors`, as in Oracle's filings) were not recognised.
- Risk differ: the "no material change" gate used a similarity ratio above
  0.98, which skipped new risk paragraphs in long sections. It now skips the
  LLM only when fewer than 5 non-numeric words changed.
- EDGAR client: bursts above 10 requests/second, ignored `Retry-After`,
  unretried transport errors, duplicate ticker-map downloads, truncated cache
  files after interrupted writes, and filings beyond the first submissions
  page being missed.
- Risk differ truncated each side to 40k characters (fabricating NEW/REMOVED
  changes) and its similarity check understated similarity of long texts.
- Engines no longer send `temperature` (rejected by current models) and skip
  malformed LLM items instead of failing; truncated or refused LLM responses
  raise clear errors.
- Ranking with `as_of` no longer scores signals published after `as_of`.
- DuckDB date filters no longer depend on the session time zone.
- Webhooks: 4xx responses are not retried, one connection pool is used per
  send, and webhook URLs (which may contain secrets) are not logged.
- The LLM client is closed after a pipeline run and filings are parsed off
  the event loop.

### Performance

- Storing 5,000 signals: 14.7 s to 0.16 s (Arrow bulk insert).
- Engines for all filings run concurrently and each MD&A is classified once.
- Section parsing makes one lxml pass over the document instead of walking
  the BeautifulSoup tree per candidate heading, with identical output: 3.57 s
  to 0.47 s for five real 10-K/10-Q filings (13.1 MB) in the offline
  benchmark.
- End-to-end, the offline benchmark pipeline (6 tickers × 6 filings, mock
  LLM) runs in 13.8 s instead of 26.7 s for 0.1.x; see `docs/benchmarks.md`.
- The risk differ's similarity check runs in a worker thread instead of
  blocking the event loop, and returns immediately for unchanged text.

### Removed

- The unused `apscheduler` dependency and a stray `mag7_signals.csv`.

## [0.1.1]

- Initial public release (published as `alphasig`, importable as `sigint`).
