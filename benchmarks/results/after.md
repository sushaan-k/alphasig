# alphasig offline benchmark results

**0.2.0 (this branch).** The suite run against the integrated code; see docs/benchmarks.md.

Mode: full

## Environment

| field | value |
|---|---|
| timestamp (UTC) | 2026-09-25T20:15:18+00:00 |
| git commit | 4f86ed04eb2a |
| code under test | 4f86ed04eb2a |
| alphasig imported from | src/alphasig |
| python | 3.11.15 (CPython) |
| platform | Linux-6.18.44-fc-v42-x86_64-with-glibc2.39 |
| cpu | Intel(R) Xeon(R) Processor @ 2.80GHz x4 |
| seed | 1234 |
| packages | alphasig 0.2.0, beautifulsoup4 4.15.0, lxml 6.1.3, duckdb 1.5.5, pyarrow 25.0.1, networkx 3.6.1, httpx 0.28.1, pydantic 2.13.5, anthropic 1.8.0, respx 0.23.1, pandas 3.0.6 |

## Parser / section extraction throughput

| input | MB | median s | MB/s | filings/s |
|---|---|---|---|---|
| real: Apple.10-K | 1.899 | 0.084 | 22.6 | 11.9 |
| real: Apple.10-Q | 1.141 | 0.041 | 27.81 | 24.38 |
| real: Nvidia.10-K | 2.669 | 0.0936 | 28.52 | 10.69 |
| real: Oracle.10-K | 5.321 | 0.2137 | 24.9 | 4.679 |
| real: Oracle.10-Q | 2.038 | 0.0797 | 25.59 | 12.55 |
| real: all 5 (5 filings) | 13.07 | 0.4715 | 27.71 | 10.6 |
| synthetic corpus (40 filings) | 25.52 | 1.664 | 15.34 | 24.05 |

_bench wall time: 19.46 s_

## Item 1A / Item 7 boundary precision & recall

| corpus | section | docs | detected | mean precision | mean recall | near-exact |
|---|---|---|---|---|---|---|
| real | risk_factors | 5 | 3 | 1 | 0.5998 | 3 |
| real | md_and_a | 5 | 5 | 1 | 0.9999 | 5 |
| synthetic | risk_factors | 50 | 50 | 1 | 1 | 50 |
| synthetic | md_and_a | 50 | 50 | 1 | 0.9999 | 50 |

Per real document (errors in characters; + = past gold end):

| document | section | precision | recall | start err | end err | gold chars |
|---|---|---|---|---|---|---|
| Apple.10-K | risk_factors | 1 | 1 | 0 | -1 | 68,759 |
| Apple.10-K | md_and_a | 1 | 0.9999 | 0 | -1 | 15,346 |
| Apple.10-Q | risk_factors | 1 | 0.9989 | 0 | -1 | 937 |
| Apple.10-Q | md_and_a | 1 | 0.9999 | 0 | -1 | 17,447 |
| Nvidia.10-K | risk_factors | 1 | 1 | 0 | -1 | 87,676 |
| Nvidia.10-K | md_and_a | 1 | 0.9995 | 18 | -1 | 38,506 |
| Oracle.10-K | risk_factors | - | - | - | - | 77,183 |
| Oracle.10-K | md_and_a | 1 | 1 | 0 | -1 | 72,169 |
| Oracle.10-Q | risk_factors | - | - | - | - | 692 |
| Oracle.10-Q | md_and_a | 1 | 1 | 0 | -1 | 60,858 |

Synthetic, by heading/TOC variant:

| variant | section | docs | detected | mean precision | mean recall |
|---|---|---|---|---|---|
| workiva | risk_factors | 10 | 10 | 1 | 1 |
| workiva | md_and_a | 10 | 10 | 1 | 0.9999 |
| inline_toc | risk_factors | 10 | 10 | 1 | 1 |
| inline_toc | md_and_a | 10 | 10 | 1 | 0.9999 |
| caps_headings | risk_factors | 10 | 10 | 1 | 1 |
| caps_headings | md_and_a | 10 | 10 | 1 | 0.9999 |
| split_heading | risk_factors | 10 | 10 | 1 | 1 |
| split_heading | md_and_a | 10 | 10 | 1 | 0.9999 |
| no_toc | risk_factors | 10 | 10 | 1 | 1 |
| no_toc | md_and_a | 10 | 10 | 1 | 0.9999 |

_bench wall time: 4.82 s_

## Risk-diff throughput

| pair | paragraphs | chars (both) | similarity | median s | chars/s |
|---|---|---|---|---|---|
| evolved | 10 | 29,964 | 0.9236 | 0.0205 | 1,458,358 |
| evolved | 40 | 119,800 | 0.9799 | 0.2781 | 430,714 |
| evolved | 80 | 235,185 | 0.9876 | 1.136 | 207,020 |
| identical | 40 | 119,642 | 1 | 0.0011 | 107,955,202 |

| filing pairs | avg section chars | LLM calls | signals | median s | pairs/s |
|---|---|---|---|---|---|
| 7 | 67,639 | 2 | 2 | 2.893 | 2.42 |

_bench wall time: 19.64 s_

## DuckDB SignalStore latency

Insert path: `arrow`

| signals | insert s | rows/s | re-insert s | ticker= ms | type+strength ms | 30d range ms | latest 1k ms | summary ms | all→Signal s | all→Arrow ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 10,000 | 0.5501 | 18,178 | 0.0814 | 1.623 | 5.882 | 2.553 | 15.24 | 4.218 | 0.212 | 13.68 |
| 100,000 | 1.424 | 70,224 | 0.8823 | 11.19 | 27.5 | 33.93 | 28.01 | 12.07 | 2.043 | 102.8 |
| 1,000,000 | 12.04 | 83,083 | 9.001 | 27.15 | 66.98 | 59.67 | 57.23 | 21.41 | 19.24 | 945.2 |

_bench wall time: 75.19 s_

## Supply-chain graph construction

| edges in | nodes | edges kept | build s | edges/s | exposure(hub) ms | top-10 ms | from signals s |
|---|---|---|---|---|---|---|---|
| 1,000 | 135 | 644 | 0.0032 | 309,483 | 0.736 | 0.161 | 0.0075 |
| 10,000 | 1,104 | 6,266 | 0.0323 | 309,438 | 6.774 | 1.663 | 0.0959 |
| 100,000 | 10,321 | 63,489 | 0.5228 | 191,294 | 81.55 | 32.87 | 1.092 |

_bench wall time: 3.3 s_

## End-to-end pipeline (mock EDGAR + latency-injected mock LLM)

6 tickers x 6 filings, mock LLM latency 0.25 s/call, mock network latency 0.02 s/request, EDGAR limiter active.

| scenario | wall s | LLM calls | peak LLM in-flight | EDGAR requests | archive fetches | signals |
|---|---|---|---|---|---|---|
| cold_default | 13.73 | 366 | 8 | 43 | 36 | 120 |
| warm_edgar_cache | 12.32 | 366 | 8 | 7 | 0 | 120 |
| cold_llm_concurrency_1 | 93.89 | 366 | 1 | 43 | 36 | 120 |
| cold_llm_concurrency_4 | 25.23 | 366 | 4 | 43 | 36 | 120 |
| cold_llm_concurrency_16 | 8.339 | 366 | 16 | 43 | 36 | 120 |
| llm_disk_cache_first_run | 15.29 | 366 | 8 | 43 | 36 | 120 |
| llm_disk_cache_rerun | 2.957 | 0 | 0 | 7 | 0 | 120 |
| incremental_first_run | 13.68 | 366 | 8 | 43 | 36 | 120 |
| incremental_rerun | 0.711 | 0 | 0 | 7 | 0 | 0 |

_bench wall time: 186.27 s_
