# alphasig offline benchmark results

**Parser before the lxml rewrite.** Parser and boundary benches run against the source at `b928903` (0.2.0 BeautifulSoup parser, before the single-pass lxml scan), harness at the commit shown.

Mode: full

## Environment

| field | value |
|---|---|
| timestamp (UTC) | 2026-09-25T20:28:05+00:00 |
| git commit | 4f86ed04eb2a |
| code under test | b928903 |
| alphasig imported from | external:baseb92/src/alphasig |
| python | 3.11.15 (CPython) |
| platform | Linux-6.18.44-fc-v42-x86_64-with-glibc2.39 |
| cpu | Intel(R) Xeon(R) Processor @ 2.80GHz x4 |
| seed | 1234 |
| packages | alphasig 0.2.0, beautifulsoup4 4.15.0, lxml 6.1.3, duckdb 1.5.5, pyarrow 25.0.1, networkx 3.6.1, httpx 0.28.1, pydantic 2.13.5, anthropic 1.8.0, respx 0.23.1, pandas 3.0.6 |

## Parser / section extraction throughput

| input | MB | median s | MB/s | filings/s |
|---|---|---|---|---|
| real: Apple.10-K | 1.899 | 0.4442 | 4.274 | 2.251 |
| real: Apple.10-Q | 1.141 | 0.256 | 4.455 | 3.906 |
| real: Nvidia.10-K | 2.669 | 0.476 | 5.607 | 2.101 |
| real: Oracle.10-K | 5.321 | 1.367 | 3.892 | 0.731 |
| real: Oracle.10-Q | 2.038 | 0.4621 | 4.412 | 2.164 |
| real: all 5 (5 filings) | 13.07 | 3.57 | 3.66 | 1.4 |
| synthetic corpus (40 filings) | 25.52 | 9.686 | 2.635 | 4.13 |

_bench wall time: 103.71 s_

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

_bench wall time: 17.75 s_
