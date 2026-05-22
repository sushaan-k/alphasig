# Signal Schema

All sigint extraction engines emit signals conforming to a universal schema. This ensures downstream consumers (backtest engines, dashboards, alert systems) only need to handle a single data structure.

## Signal Fields

| Field | Type | Description |
|---|---|---|
| `timestamp` | `datetime` (UTC) | Filing date |
| `ticker` | `str` | Company ticker symbol |
| `signal_type` | `enum` | One of: `supply_chain`, `risk_change`, `m_and_a`, `tone_shift` |
| `direction` | `enum` | `bullish`, `bearish`, or `neutral` |
| `strength` | `float` | Signal strength, 0.0 to 1.0 |
| `confidence` | `float` | Extraction confidence, 0.0 to 1.0 |
| `context` | `str` | Human-readable explanation |
| `source_filing` | `str` | EDGAR filing URL |
| `related_tickers` | `list[str]` | Other affected companies |
| `metadata` | `dict` | Engine-specific details |

## Signal Types

### `supply_chain`
Metadata includes: `target` (supplier/partner name), `relation` (depends_on, supplies_to, partners_with), `edge_context`.

### `risk_change`
Metadata includes: `change_type` (NEW, REMOVED, ESCALATED, DE_ESCALATED), `severity` (LOW, MEDIUM, HIGH, CRITICAL), `language_shift`, `current_filing`, `previous_filing`.

### `m_and_a`
Metadata includes: `indicator_count`, `categories` (list of detected M&A categories), `indicators` (detailed list).

### `tone_shift`
Metadata includes: `topic`, `current_tone`, `previous_tone`, `key_phrases`.

## Output Formats

Signals can be exported to:
- **Parquet** -- `signals.to_parquet("path.parquet")`
- **CSV** -- `signals.to_csv("path.csv")`
- **REST API** -- `signals.to_api(port=8080)`
- **DuckDB** -- Automatic via `SignalStore`

## Ranking Reports

Stored signals can also be transformed into ticker-level ranking reports with
`sigint rank` or `rank_signals(signals)`. Each ticker score includes:

| Field | Description |
|---|---|
| `score` | Average confidence-weighted directional strength, negative for bearish exposure |
| `gross_score` | Average confidence-weighted absolute exposure |
| `signal_count` | Number of signals contributing to the ticker |
| `avg_strength` | Average effective strength, including decay when an `as_of` timestamp is used |
| `avg_confidence` | Average extraction confidence |
| `top_contexts` | Highest-weight signal explanations for analyst review |
| `related_tickers` | Count of related tickers mentioned by contributing signals |

The report renders as JSON or Markdown for CI artifacts, research notebooks, or
portfolio review notes.
