# sigint

## LLM-Powered Causal Signal Extraction from SEC Filings

### The Problem

Every quant fund scrapes SEC filings. The basic approach — sentiment analysis on 10-K/10-Q text — is a solved (and commoditized) problem. FinBERT can do it. GPT-4 can do it. There's zero alpha left in "is this filing positive or negative?"

What **hasn't** been done is **causal information extraction** — understanding the *structural relationships* buried in filings:
- Which companies are supply chain dependencies of which others?
- How did risk factors change between this quarter and last?
- What M&A language signals an acquisition is coming?
- When management tone shifts on a specific topic, what happens to the stock?

This is the difference between "sentiment is positive" (useless) and "Company X just added 'supply chain concentration risk' to their 10-K for the first time, and their top supplier is Company Y which reports next week" (actionable).

arXiv research from March 2026 (FinToolBench) showed that LLM agents still struggle with real financial tool use — the tooling doesn't exist for structured extraction at this depth.

### The Solution

`sigint` is a pipeline that ingests SEC filings (via EDGAR), performs deep causal and structural extraction using LLMs, and outputs structured, backtestable signals.

### Pipeline Architecture

```
┌──────────────────────────────────────────────────────────┐
│                        sigint                             │
│                                                           │
│  ┌──────────────┐                                         │
│  │  EDGAR       │  Pulls 10-K, 10-Q, 8-K, DEF 14A        │
│  │  Ingestion   │  filings. Handles rate limits, caching,  │
│  │              │  incremental updates.                    │
│  └──────┬───────┘                                         │
│         │                                                  │
│         ▼                                                  │
│  ┌──────────────┐                                         │
│  │  Section     │  Parses XBRL/HTML into structured        │
│  │  Parser      │  sections: Risk Factors, MD&A,           │
│  │              │  Notes to Financial Statements, etc.     │
│  └──────┬───────┘                                         │
│         │                                                  │
│         ▼                                                  │
│  ┌──────────────────────────────────────────────────┐     │
│  │              Extraction Engines                    │     │
│  │                                                    │     │
│  │  ┌──────────────┐  ┌──────────────────────────┐   │     │
│  │  │ Supply Chain  │  │  Risk Factor Differ      │   │     │
│  │  │ Graph Builder │  │                          │   │     │
│  │  │              │  │  Diffs risk factors       │   │     │
│  │  │  Extracts    │  │  between consecutive      │   │     │
│  │  │  supplier/   │  │  filings. Classifies:     │   │     │
│  │  │  customer    │  │  - New risks              │   │     │
│  │  │  mentions,   │  │  - Removed risks          │   │     │
│  │  │  builds      │  │  - Escalated language     │   │     │
│  │  │  dependency  │  │  - De-escalated language  │   │     │
│  │  │  graph       │  │                          │   │     │
│  │  └──────────────┘  └──────────────────────────┘   │     │
│  │                                                    │     │
│  │  ┌──────────────┐  ┌──────────────────────────┐   │     │
│  │  │ M&A Signal   │  │  Management Tone         │   │     │
│  │  │ Detector     │  │  Analyzer                │   │     │
│  │  │              │  │                          │   │     │
│  │  │  Identifies  │  │  Tracks topic-level      │   │     │
│  │  │  acquisition │  │  tone shifts across      │   │     │
│  │  │  language,   │  │  filings. Not just       │   │     │
│  │  │  merger      │  │  "positive/negative"     │   │     │
│  │  │  signals,    │  │  but "confident →        │   │     │
│  │  │  divestiture │  │  hedging" on specific    │   │     │
│  │  │  hints       │  │  topics like revenue     │   │     │
│  │  │              │  │  growth or margins       │   │     │
│  │  └──────────────┘  └──────────────────────────┘   │     │
│  └──────────────────────────────────────────────────┘     │
│         │                                                  │
│         ▼                                                  │
│  ┌──────────────┐                                         │
│  │  Signal       │  Converts extracted information into    │
│  │  Compiler     │  structured, timestamped signals        │
│  │              │  suitable for backtesting.               │
│  └──────┬───────┘                                         │
│         │                                                  │
│         ▼                                                  │
│  ┌──────────────┐                                         │
│  │  Output       │  Parquet files, REST API, or           │
│  │  Formats      │  webhook notifications.                │
│  └──────────────┘                                         │
└──────────────────────────────────────────────────────────┘
```

### Extraction Engines (Detail)

#### 1. Supply Chain Graph Builder

Most companies mention key suppliers and customers in their 10-K (often required by SEC rules). `sigint` extracts these into a **knowledge graph**:

```python
# Output: supply chain edges
[
    Edge(source="AAPL", target="TSMC", relation="depends_on",
         context="semiconductor manufacturing", confidence=0.95,
         filing="10-K", date="2025-11-01"),
    Edge(source="AAPL", target="Foxconn", relation="depends_on",
         context="device assembly", confidence=0.92,
         filing="10-K", date="2025-11-01"),
]
```

Why this matters: When TSMC reports earnings or faces a supply disruption, you know exactly which companies are exposed. This graph doesn't exist in any commercial data product at filing-level granularity.

#### 2. Risk Factor Differ

10-K/10-Q filings have a "Risk Factors" section. Companies are legally required to disclose material risks. `sigint` diffs these between consecutive filings:

```python
# Output: risk factor changes
[
    RiskChange(
        company="MSFT",
        type="NEW",
        risk="Concentration of AI infrastructure spending",
        section="Item 1A",
        current_filing="10-K 2025",
        previous_filing="10-K 2024",
        severity_estimate="HIGH",
        related_tickers=["NVDA", "AMD"],
    ),
    RiskChange(
        company="MSFT",
        type="ESCALATED",
        risk="Regulatory scrutiny of AI products",
        language_shift="'may face' → 'are currently subject to'",
        severity_estimate="CRITICAL",
    ),
]
```

#### 3. M&A Signal Detector

Certain language patterns in filings strongly predict upcoming M&A activity:
- New mentions of "strategic alternatives," "potential transactions," "advisors"
- Changes in cash reserves commentary
- New board members with M&A backgrounds
- Unusual related-party transaction disclosures

The LLM is prompted with M&A-specific extraction schemas trained on historical confirmed-M&A filings.

#### 4. Management Tone Analyzer

Goes beyond positive/negative. Tracks **topic-specific tone trajectories**:

```python
# Output: tone trajectory
ToneTrajectory(
    company="META",
    topic="AI infrastructure spending",
    trajectory=[
        TonePoint("10-Q Q1 2025", "confident_expanding", 0.85),
        TonePoint("10-Q Q2 2025", "confident_expanding", 0.82),
        TonePoint("10-Q Q3 2025", "hedging_cautious", 0.71),  # ← shift
        TonePoint("10-K 2025", "defensive_justifying", 0.65),  # ← escalation
    ],
    signal="BEARISH_SHIFT",
    signal_strength=0.78,
)
```

### Signal Output Schema

All signals follow a standard schema for backtesting compatibility:

```python
@dataclass
class Signal:
    timestamp: datetime          # Filing date
    ticker: str                  # Company ticker
    signal_type: str             # "supply_chain" | "risk_change" | "m_and_a" | "tone_shift"
    direction: str               # "bullish" | "bearish" | "neutral"
    strength: float              # 0.0 - 1.0
    confidence: float            # 0.0 - 1.0
    context: str                 # Human-readable explanation
    source_filing: str           # EDGAR filing URL
    related_tickers: list[str]   # Other affected companies
    metadata: dict               # Engine-specific details
```

Output formats:
- **Parquet** (for backtest engines like Lean, Zipline)
- **REST API** (for live systems)
- **Webhooks** (for alerts)
- **CSV** (for Excel warriors)

### Technical Stack

- **Language**: Python 3.11+
- **EDGAR access**: `sec-edgar-downloader` + custom rate-limited client
- **Parsing**: `beautifulsoup4` for HTML filings, `python-xbrl` for XBRL
- **LLM**: Any model via API (Anthropic recommended for long-context)
- **Graph**: `networkx` for supply chain graphs
- **Storage**: `duckdb` for local, `parquet` for file output
- **Scheduling**: `apscheduler` for daily EDGAR polling

### API Surface (Draft)

```python
from sigint import Pipeline, EDGAR, Signals

# Initialize
edgar = EDGAR(api_key="...", cache_dir="./edgar_cache")
pipeline = Pipeline(model="claude-sonnet-4-6")

# Extract signals from a specific company
signals = await pipeline.extract(
    tickers=["AAPL", "MSFT", "GOOGL", "META", "AMZN"],
    filing_types=["10-K", "10-Q"],
    lookback_years=3,
    engines=["supply_chain", "risk_differ", "m_and_a", "tone"],
)

# Get supply chain graph
graph = signals.supply_chain_graph()
graph.plot()
graph.exposure("TSMC")  # Which companies depend on TSMC?

# Get risk factor changes
changes = signals.risk_changes(severity="HIGH")

# Export for backtesting
signals.to_parquet("signals.parquet")
signals.to_api(port=8080)
```

### What Makes This Novel

1. **Causal extraction, not sentiment** — structurally different from every other NLP-on-filings tool
2. **Supply chain graph from filings** — this data product doesn't exist commercially at this granularity
3. **Risk factor diffing** — legal language changes are some of the strongest predictive signals
4. **Directly ties to M&A research** — extends your UTampa research into code
5. **Backtestable output** — not just analysis, but structured signals you can trade on

### Repo Structure

```
sigint/
├── README.md
├── pyproject.toml
├── src/
│   └── sigint/
│       ├── __init__.py
│       ├── edgar.py            # EDGAR API client
│       ├── parser.py           # Filing section parser
│       ├── engines/
│       │   ├── supply_chain.py # Supply chain graph extraction
│       │   ├── risk_differ.py  # Risk factor diffing
│       │   ├── m_and_a.py      # M&A signal detection
│       │   └── tone.py         # Management tone analysis
│       ├── signals.py          # Signal schema and compilation
│       ├── graph.py            # Supply chain graph operations
│       ├── output/
│       │   ├── parquet.py      # Parquet export
│       │   ├── api.py          # REST API server
│       │   └── webhook.py      # Webhook notifications
│       └── pipeline.py         # Main orchestration
├── tests/
├── examples/
│   ├── mag7_analysis.py        # Analyze Magnificent 7
│   ├── supply_chain_map.py     # Visualize supply chains
│   └── risk_monitor.py         # Monitor risk factor changes
└── docs/
    ├── engines.md
    ├── signal_schema.md
    └── backtesting.md
```

### Research References

- "FinToolBench: Benchmarking LLM Agents with Real-World Financial Tools" (arXiv:2603.08262, Mar 2026)
- "From Deep Learning to LLMs: A Survey of AI in Quantitative Investment" (arXiv:2503.21422, Mar 2026)
- "The Inference Premium: Why 2026 is the Year LLM Logic Overtook Quantitative Statistics" (AInvest)
- SEC EDGAR Full-Text Search API documentation
- "Lazy Prices" (Cohen, Malloy, Nguyen, 2020) — seminal paper on 10-K language changes predicting returns
