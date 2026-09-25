# alphasig offline benchmark results

**0.1.x baseline.** The suite run against the source at `54efd36` (the 0.1.x code base just renamed to `alphasig`, before the 0.2.0 fixes and optimisations), with the harness from revision `9def475`. Measurements are unchanged from that run.

Mode: full

## Environment

| field | value |
|---|---|
| timestamp (UTC) | 2026-09-25T19:09:19+00:00 |
| git commit | 9def475896b6 (dirty) |
| code under test | 54efd364b7b3 |
| alphasig imported from | external:baseline_src/src/alphasig |
| python | 3.11.15 (CPython) |
| platform | Linux-6.18.44-fc-v42-x86_64-with-glibc2.39 |
| cpu | Intel(R) Xeon(R) Processor @ 2.80GHz x4 |
| seed | 1234 |
| packages | alphasig 0.2.0, beautifulsoup4 4.15.0, lxml 6.1.3, duckdb 1.5.5, pyarrow 25.0.1, networkx 3.6.1, httpx 0.28.1, pydantic 2.13.5, anthropic 0.125.0, respx 0.23.1 |

## Parser / section extraction throughput

| input | MB | median s | MB/s | filings/s |
|---|---|---|---|---|
| real: Apple.10-K | 1.899 | 0.4079 | 4.654 | 2.452 |
| real: Apple.10-Q | 1.141 | 0.288 | 3.961 | 3.472 |
| real: Nvidia.10-K | 2.669 | 0.474 | 5.631 | 2.11 |
| real: Oracle.10-K | 5.321 | 1.328 | 4.008 | 0.753 |
| real: Oracle.10-Q | 2.038 | 0.4273 | 4.771 | 2.34 |
| real: all 5 (5 filings) | 13.07 | 3.659 | 3.572 | 1.367 |
| synthetic corpus (40 filings) | 25.52 | 9.928 | 2.571 | 4.029 |

_bench wall time: 105.54 s_

## Item 1A / Item 7 boundary precision & recall

| corpus | section | docs | detected | mean precision | mean recall | near-exact |
|---|---|---|---|---|---|---|
| real | risk_factors | 5 | 4 | 0.7476 | 0.8 | 1 |
| real | md_and_a | 5 | 5 | 0.7981 | 0.9999 | 1 |
| synthetic | risk_factors | 50 | 43 | 0.917 | 0.86 | 0 |
| synthetic | md_and_a | 50 | 50 | 0.7731 | 0.8 | 0 |

Per real document (errors in characters; + = past gold end):

| document | section | precision | recall | start err | end err | gold chars |
|---|---|---|---|---|---|---|
| Apple.10-K | risk_factors | 0.9612 | 1 | 0 | 2,777 | 68,759 |
| Apple.10-K | md_and_a | 0.8356 | 1 | 0 | 3,020 | 15,346 |
| Apple.10-Q | risk_factors | 0.2169 | 1 | 0 | 3,383 | 937 |
| Apple.10-Q | md_and_a | 0.9781 | 1 | 0 | 390 | 17,447 |
| Nvidia.10-K | risk_factors | 0.9994 | 1 | 0 | 50 | 87,676 |
| Nvidia.10-K | md_and_a | 0.8921 | 0.9995 | 18 | 4,653 | 38,506 |
| Oracle.10-K | risk_factors | 0.8128 | 1 | -4,538 | 13,235 | 77,183 |
| Oracle.10-K | md_and_a | 0.2917 | 1 | 0 | 175,217 | 72,169 |
| Oracle.10-Q | risk_factors | - | - | - | - | 692 |
| Oracle.10-Q | md_and_a | 0.9927 | 1 | 0 | 447 | 60,858 |

Synthetic, by heading/TOC variant:

| variant | section | docs | detected | mean precision | mean recall |
|---|---|---|---|---|---|
| workiva | risk_factors | 10 | 10 | 0.9628 | 1 |
| workiva | md_and_a | 10 | 10 | 0.9662 | 1 |
| inline_toc | risk_factors | 10 | 3 | 0.3159 | 0.3 |
| inline_toc | md_and_a | 10 | 10 | 0 | 0 |
| caps_headings | risk_factors | 10 | 10 | 0.9616 | 1 |
| caps_headings | md_and_a | 10 | 10 | 0.9667 | 1 |
| split_heading | risk_factors | 10 | 10 | 0.9633 | 1 |
| split_heading | md_and_a | 10 | 10 | 0.9674 | 1 |
| no_toc | risk_factors | 10 | 10 | 0.9606 | 1 |
| no_toc | md_and_a | 10 | 10 | 0.965 | 1 |

_bench wall time: 19.07 s_

## Risk-diff throughput

| pair | paragraphs | chars (both) | similarity | median s | chars/s |
|---|---|---|---|---|---|
| evolved | 10 | 29,964 | 0.4885 | 0.0409 | 733,293 |
| evolved | 40 | 119,800 | 0.2999 | 2.465 | 48,599 |
| evolved | 80 | 235,185 | 0.2867 | 14.06 | 16,725 |
| identical | 40 | 119,642 | 1 | 0.2162 | 553,442 |

| filing pairs | avg section chars | LLM calls | signals | median s | pairs/s |
|---|---|---|---|---|---|
| 7 | 67,639 | 7 | 7 | 19.19 | 0.365 |

_bench wall time: 165.97 s_

## DuckDB SignalStore latency

Insert path: `executemany`

| signals | insert s | rows/s | re-insert s | ticker= ms | type+strength ms | 30d range ms | latest 1k ms | summary ms | all→Signal s | all→Arrow ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 10,000 | 57.17 | 175 | - | 2.744 | 10.55 | 9.166 | 17.97 | 5.543 | 0.19 | - |
| 100,000 | skipped: row-by-row insert too slow | - | - | - | - | - | - | - | - | - |
| 1,000,000 | skipped: row-by-row insert too slow | - | - | - | - | - | - | - | - | - |

_bench wall time: 58.1 s_

## Supply-chain graph construction

| edges in | nodes | edges kept | build s | edges/s | exposure(hub) ms | top-10 ms | from signals s |
|---|---|---|---|---|---|---|---|
| 1,000 | 135 | 644 | 0.0035 | 288,695 | 0.777 | 0.142 | 0.0067 |
| 10,000 | 1,104 | 6,266 | 0.0523 | 191,102 | 11.55 | 3.846 | 0.2147 |
| 100,000 | 10,321 | 63,489 | 0.4765 | 209,845 | 90.11 | 46.3 | 1.68 |

_bench wall time: 5.61 s_

## End-to-end pipeline (mock EDGAR + latency-injected mock LLM)

6 tickers x 6 filings, mock LLM latency 0.25 s/call, mock network latency 0.02 s/request, EDGAR limiter active.

| scenario | wall s | LLM calls | peak LLM in-flight | EDGAR requests | archive fetches | signals |
|---|---|---|---|---|---|---|
| cold_default | 28.86 | 372 | 12 | 45 | 36 | 120 |
| warm_edgar_cache | 27.09 | 372 | 12 | 9 | 0 | 120 |
| cold_llm_concurrency_* | skipped: Pipeline has no llm_concurrency option | - | - | - | - | - | - |
| llm_disk_cache_* | skipped: Pipeline has no llm_cache_dir option | - | - | - | - | - | - |
| incremental_* | skipped: Pipeline.extract has no incremental option | - | - | - | - | - | - |

_bench wall time: 56.03 s_
