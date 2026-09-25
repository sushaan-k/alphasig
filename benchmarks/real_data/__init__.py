"""Real-data evaluation harness (scaffold -- not run in CI, no results committed).

These scripts need things the offline suite deliberately avoids: live
EDGAR access (a real ``ALPHASIG_USER_AGENT``), a paid LLM (``ANTHROPIC_API_KEY``)
and, for the event study, a price file you supply.  They refuse to run
without them.  See ``docs/benchmarks.md`` ("Real-data harness").
"""
