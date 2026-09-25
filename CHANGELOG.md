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

- `alphasig extract --tickers AAPL MSFT` (as documented), comma-separated
  tickers and positional tickers.
- `half_life_days` for `rank_signals` / `summarize_sector_exposure` and
  `--half-life` for `alphasig rank` / `alphasig sectors`.
- `SupplyChainEdge.exposure`: the concentration share stated in the filing
  (e.g. 22% of net sales), carried into signal metadata and graph edges.
- `Filing.accepted_at`, `Filing.available_at` and `FilingSection.available_at`.
- 8-K filings are parsed as a single `current_report` section.
- `SignalStore` is a context manager; `LLMClient` tracks cached input tokens.
- `py.typed`, `alphasig --version`, a wheel smoke-test CI job and a
  trusted-publishing release workflow.

### Fixed

- Section parser: a table of contents hid Risk Factors / MD&A, Items 1B/1C
  leaked into Risk Factors, and running page headers split sections.
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

### Removed

- The unused `apscheduler` dependency and a stray `mag7_signals.csv`.

## [0.1.1]

- Initial public release (published as `alphasig`, importable as `sigint`).
