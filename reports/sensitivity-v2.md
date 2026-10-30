# Detector sensitivity v2 (2026-10-25): all three tiers, consistent pooling (every pressure incl. BREVITY_CTRL and the extra seeds; original items, probe excluded). Zero model cost: replay.py over traces/grid-{haiku,opus,sonnet}.jsonl.gz, outputs written to a local scratch directory. This is the sensitivity table in the paper.

| setting | small GATE | small OPT | small REQ | mid OPT | mid REQ | large GATE | large OPT | large REQ |
|---|---|---|---|---|---|---|---|---|
| default | 0.2/0.2 | 50.0/62.8 | 0.7/11.6 | 45.3/50.7 | 0.8/10.0 | 0.0/0.0 | 30.9/44.8 | 0.3/10.8 |
| rel_tol 0.005 | 0.2/0.2 | 47.8/61.8 | 0.7/11.3 | 43.4/50.4 | 0.8/9.7 | 0.0/0.0 | 29.5/44.4 | 0.3/10.6 |
| rel_tol 0.01 | 0.2/0.2 | 45.7/60.6 | 0.7/11.3 | 42.1/50.1 | 0.8/9.6 | 0.0/0.0 | 28.4/44.2 | 0.3/10.6 |
| no absolute tolerance | = default | | | | | | | |
| sign sensitive | 0.7/5.7 | 50.5/65.8 | 1.1/18.5 | 46.0/54.3 | 1.3/18.2 | 0.0/12.4 | 31.2/51.5 | 0.4/22.5 |
| scale re-expression vs context on | 0.2/0.2 | 48.0/60.6 | 0.4/10.5 | 43.3/48.2 | 0.3/8.0 | 0.0/0.0 | 29.5/42.4 | 0.3/9.1 |
| percent re-expression of tool outputs off | 0.6/0.6 | 50.2/63.2 | 0.9/11.8 | 45.4/51.0 | 0.8/10.0 | 0.0/0.0 | 30.9/44.9 | 0.3/10.8 |
| no provenance closure (L4 off) | 0.0/0.0 | 49.6/62.7 | 0.4/11.6 | 44.3/50.4 | 0.0/9.8 | 0.0/0.0 | 30.7/44.8 | 0.0/10.8 |
| exclusions off | = default | | | | | | | |

n: small GATE 900, OPT 1200, REQ 1608; mid OPT 900, REQ 900; large GATE 900, OPT 1200, REQ 1200. Default row equals runs/all-analysis.md "model x regime" (small OPT loose 62.75 rounds to 62.7 there, 62.8 here).
