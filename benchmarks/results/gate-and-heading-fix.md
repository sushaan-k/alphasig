# alphasig offline benchmark results

after risk-diff changed-words gate and split-initial heading fix

Mode: full

## Environment

| field | value |
|---|---|
| timestamp (UTC) | 2026-09-25T20:38:59+00:00 |
| git commit | 8cbc6038aa6c (dirty) |
| code under test | 8cbc6038aa6c |
| alphasig imported from | src/alphasig |
| python | 3.11.15 (CPython) |
| platform | Linux-6.18.44-fc-v42-x86_64-with-glibc2.39 |
| cpu | Intel(R) Xeon(R) Processor @ 2.80GHz x4 |
| seed | 1234 |
| packages | alphasig 0.2.0, beautifulsoup4 4.15.0, lxml 6.1.3, duckdb 1.5.5, pyarrow 25.0.1, networkx 3.6.1, httpx 0.28.1, pydantic 2.13.5, anthropic 1.8.0, respx 0.23.1, pandas 3.0.6 |

## Item 1A / Item 7 boundary precision & recall

| corpus | section | docs | detected | mean precision | mean recall | near-exact |
|---|---|---|---|---|---|---|
| real | risk_factors | 5 | 5 | 1 | 0.9995 | 5 |
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
| Oracle.10-K | risk_factors | 1 | 1 | 0 | -1 | 77,183 |
| Oracle.10-K | md_and_a | 1 | 1 | 0 | -1 | 72,169 |
| Oracle.10-Q | risk_factors | 1 | 0.9986 | 0 | -1 | 692 |
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

_bench wall time: 5.0 s_

## Risk-diff throughput

| pair | paragraphs | chars (both) | similarity | median s | chars/s |
|---|---|---|---|---|---|
| evolved | 10 | 29,964 | 0.9236 | 0.0193 | 1,551,663 |
| evolved | 40 | 119,800 | 0.9799 | 0.2828 | 423,564 |
| evolved | 80 | 235,185 | 0.9876 | 1.099 | 213,958 |
| identical | 40 | 119,642 | 1 | 0.0012 | 102,050,965 |

| filing pairs | avg section chars | LLM calls | signals | median s | pairs/s |
|---|---|---|---|---|---|
| 7 | 67,639 | 7 | 7 | 2.77 | 2.527 |

_bench wall time: 19.36 s_
