# alphasig

[![CI](https://github.com/sushaan-k/alphasig/actions/workflows/ci.yml/badge.svg)](https://github.com/sushaan-k/alphasig/actions)
[![PyPI](https://img.shields.io/pypi/v/alphasig.svg)](https://pypi.org/project/alphasig/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/alphasig.svg)](https://pypi.org/project/alphasig/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Causal signal extraction from SEC EDGAR filings — not correlation, actual causation.**

`alphasig` pulls structured data from SEC EDGAR, parses 10-K/10-Q risk factors and footnotes, extracts causal relationships using NLP, builds a knowledge graph of corporate dependencies, and surfaces backtestable trading signals based on *causes* rather than lagged price moves.

---

## The Problem

Most quant signal extraction from text is pure NLP correlation: sentiment scores, topic models, keyword frequencies. These signals are noisy, crowded, and decay fast. The actual alpha is in the *structure* of what companies say — which suppliers they depend on, which regulatory changes they flag as risks, which competitors they name. This is causal information buried in 200-page filings that nobody is systematically extracting.

## Solution

```python
from sigint import Pipeline, SignalConfig

pipe = Pipeline(
    tickers=["NVDA", "AMD", "INTC", "TSMC", "ASML"],
    filing_types=["10-K", "10-Q"],
    lookback_years=5,
)

graph = await pipe.build_causal_graph()

# Which companies upstream of NVDA are flagged as single-source risks?
upstream_risks = graph.causal_paths("NVDA", relation="supply_dependency", depth=3)

signals = pipe.extract_signals(graph, config=SignalConfig(
    causal_threshold=0.7,
    require_backtestable=True,
))

for sig in signals.top(10):
    print(f"{sig.ticker} | {sig.direction} | cause: {sig.causal_chain}")
# ASML | LONG | cause: TSMC→NVDA supply dependency flagged as resolved in Q3 10-Q
```

## At a Glance

- **Direct EDGAR access** — no paid data vendor required
- **Causal NLP** — extracts `causes`, `depends_on`, `risks`, `resolves` relations (not just sentiment)
- **Knowledge graph** — corporate dependency graph queryable by path, depth, and relation type
- **Backtestable signals** — every signal includes entry logic, catalyst event, and historical hit rate
- **Scheduler** — auto-pulls new filings and re-runs signal extraction on your cadence

## Install

```bash
pip install alphasig
```

## Signal Types

| Signal Class | Description | Typical Edge |
|---|---|---|
| Supply Chain Concentration | Single-source risk newly flagged or resolved | 3–8 day drift |
| Regulatory Catalyst | New rule flagged in risk factors, affects sector | Event-driven |
| Competitor Mention Shift | Competitor newly named or dropped from risk section | 1–5 day mean reversion |
| Guidance Language Delta | Causal language change in forward-looking statements | Earnings window |

## Architecture

```
Pipeline
 ├── EdgarClient       # async EDGAR full-text search + filing download
 ├── FilingParser      # extracts structured sections (risk factors, MD&A, footnotes)
 ├── CausalExtractor   # NLP causal relation extraction
 ├── KnowledgeGraph    # networkx-backed corporate dependency graph
 └── SignalEngine      # backtests and ranks candidate signals
```

## Contributing

PRs welcome. Run `pip install -e ".[dev]"` then `pytest`. Star the repo if you find it useful ⭐
