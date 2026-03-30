# Extraction Engines

sigint ships with four extraction engines, each targeting a different class of causal signal buried in SEC filings.

## Supply Chain Graph Builder

**Engine name:** `supply_chain`

Reads the Business and Risk Factors sections of 10-K/10-Q filings and uses the LLM to identify supplier, customer, and partner relationships. Outputs a directed knowledge graph.

**Sections used:** Business (Item 1), Risk Factors (Item 1A), MD&A (Item 7)

**Output:** `SupplyChainEdge` instances with source, target, relation type, context, and confidence.

**Signal direction:** Always `neutral` (the relationship itself is informational; downstream analysis determines direction based on events affecting the supplier).

## Risk Factor Differ

**Engine name:** `risk_differ`

Compares the Risk Factors section (Item 1A) between consecutive filings for the same company. Classifies each material change as NEW, REMOVED, ESCALATED, or DE_ESCALATED.

Inspired by the "Lazy Prices" paper (Cohen, Malloy, Nguyen 2020) which demonstrated that 10-K language changes are among the strongest predictors of future returns.

**Sections used:** Risk Factors (Item 1A)

**Requires previous filing:** Yes

**Signal direction:** NEW/ESCALATED = bearish, REMOVED/DE_ESCALATED = bullish.

**Severity levels:** LOW, MEDIUM, HIGH, CRITICAL

## M&A Signal Detector

**Engine name:** `m_and_a`

Scans all filing sections for language patterns that historically precede M&A activity:

- **Strategic alternatives** -- "exploring strategic alternatives", "potential transactions"
- **Advisor engagement** -- New mentions of investment banks or legal counsel
- **Cash positioning** -- Unusual cash reserves commentary or new credit facilities
- **Board changes** -- Directors with M&A/PE backgrounds
- **Related-party transactions** -- Unusual disclosures that may signal insider deal activity

**Sections used:** All available sections

**Signal direction:** Typically `bullish` when strategic-alternatives language is strong (implies target premium), otherwise `neutral`.

## Management Tone Analyzer

**Engine name:** `tone`

Goes beyond simple positive/negative polarity to track topic-specific tone trajectories across filings. For each topic (e.g. "revenue growth", "AI spending"), classifies management's tone along a six-point scale:

1. `confident_expanding` -- Strong conviction, expansion language
2. `optimistic_cautious` -- Positive but measured
3. `neutral_factual` -- Purely factual
4. `hedging_cautious` -- Qualifiers, hedging language
5. `defensive_justifying` -- Defending past decisions
6. `pessimistic_warning` -- Explicit warnings

**Sections used:** MD&A (Item 7/Item 2)

**Requires previous filing:** Yes (for shift detection)

**Signal direction:** Determined by the direction and magnitude of the tone shift.
