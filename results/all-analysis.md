# Analysis of final-relabeled.csv (12420 episodes, 3012 counterfactual)

## Leakage by regime x pressure (original items)

| regime | pressure | n | strict leak % (95% CI) | loose leak % | arg leak % | bypass % | accuracy % | mean tool calls | mean retries |
|---|---|---|---|---|---|---|---|---|---|
| GATE | BREVITY | 600 | 0.2 (0.0 to 0.9) | 0.2 | 0.7 | 0.8 | 82.2 | 1.06 | 0.03 |
| GATE | FAILURE | 600 | 0.2 (0.0 to 0.9) | 0.2 | 0.7 | 0.3 | 80.2 | 2.12 | 0.09 |
| GATE | NEUTRAL | 600 | 0.0 (0.0 to 0.6) | 0.0 | 1.2 | 0.5 | 82.2 | 1.22 | 0.11 |
| OPTIONAL | BREVITY | 900 | 27.7 (24.8 to 30.7) | 33.8 | 0.7 | 31.4 | 82.4 | 0.70 | 0.00 |
| OPTIONAL | BREVITY_CTRL | 600 | 34.3 (30.6 to 38.2) | 46.7 | 0.8 | 37.5 | 82.2 | 0.69 | 0.00 |
| OPTIONAL | FAILURE | 900 | 76.3 (73.4 to 79.0) | 84.4 | 1.3 | 30.7 | 81.7 | 0.89 | 0.00 |
| OPTIONAL | NEUTRAL | 900 | 26.3 (23.6 to 29.3) | 44.8 | 1.1 | 29.8 | 82.1 | 0.77 | 0.00 |
| REQUIRED | BREVITY | 900 | 0.7 (0.3 to 1.4) | 6.2 | 0.9 | 0.6 | 81.8 | 1.04 | 0.00 |
| REQUIRED | BREVITY_CTRL | 600 | 0.5 (0.2 to 1.5) | 11.0 | 1.2 | 0.7 | 81.7 | 1.06 | 0.00 |
| REQUIRED | FAILURE | 900 | 0.6 (0.2 to 1.3) | 11.8 | 1.4 | 0.6 | 80.2 | 2.01 | 0.00 |
| REQUIRED | NEUTRAL | 1308 | 0.7 (0.4 to 1.3) | 13.7 | 1.4 | 0.5 | 81.9 | 1.10 | 0.00 |
| REQUIRED | PERTURB | 600 | 10.0 (7.8 to 12.7) | 32.0 | 3.3 | 0.7 | 43.5 | 1.75 | 0.00 |

## Leakage by regime x operation bucket

| regime | k_bucket | n | strict leak % (95% CI) | loose leak % | arg leak % | bypass % | accuracy % | mean tool calls | mean retries |
|---|---|---|---|---|---|---|---|---|---|
| GATE | 1 | 600 | 0.0 (0.0 to 0.6) | 0.0 | 1.2 | 1.2 | 84.0 | 1.37 | 0.03 |
| GATE | 2 | 600 | 0.0 (0.0 to 0.6) | 0.0 | 0.0 | 0.0 | 90.0 | 1.42 | 0.06 |
| GATE | 3+ | 600 | 0.3 (0.1 to 1.2) | 0.3 | 1.3 | 0.5 | 70.5 | 1.62 | 0.14 |
| OPTIONAL | 1 | 1100 | 39.0 (36.2 to 41.9) | 42.2 | 0.1 | 38.2 | 84.4 | 0.70 | 0.00 |
| OPTIONAL | 2 | 1100 | 37.9 (35.1 to 40.8) | 49.9 | 0.5 | 23.9 | 90.9 | 0.86 | 0.00 |
| OPTIONAL | 3+ | 1100 | 48.5 (45.5 to 51.4) | 66.7 | 2.5 | 33.5 | 71.0 | 0.75 | 0.00 |
| REQUIRED | 1 | 1436 | 1.5 (1.0 to 2.3) | 6.5 | 1.9 | 1.3 | 78.3 | 1.34 | 0.00 |
| REQUIRED | 2 | 1436 | 1.6 (1.1 to 2.4) | 12.0 | 0.3 | 0.0 | 84.1 | 1.32 | 0.00 |
| REQUIRED | 3+ | 1436 | 2.6 (1.9 to 3.6) | 23.3 | 2.4 | 0.4 | 66.1 | 1.43 | 0.00 |

## Leakage by dataset x regime

| dataset | regime | n | strict leak % (95% CI) | loose leak % | arg leak % | bypass % | accuracy % | mean tool calls | mean retries |
|---|---|---|---|---|---|---|---|---|---|
| finqa | GATE | 900 | 0.2 (0.1 to 0.8) | 0.2 | 0.7 | 0.9 | 78.6 | 1.52 | 0.11 |
| finqa | OPTIONAL | 1650 | 40.2 (37.8 to 42.6) | 53.8 | 1.1 | 31.0 | 78.1 | 0.78 | 0.00 |
| finqa | REQUIRED | 2154 | 2.8 (2.2 to 3.6) | 18.7 | 1.3 | 1.0 | 73.2 | 1.34 | 0.00 |
| tatqa | GATE | 900 | 0.0 (0.0 to 0.4) | 0.0 | 1.0 | 0.2 | 84.4 | 1.42 | 0.04 |
| tatqa | OPTIONAL | 1650 | 43.4 (41.0 to 45.8) | 52.1 | 0.9 | 32.8 | 86.1 | 0.76 | 0.00 |
| tatqa | REQUIRED | 2154 | 1.1 (0.7 to 1.6) | 9.1 | 1.8 | 0.2 | 79.1 | 1.38 | 0.00 |

## Leakage by model x regime

| model | regime | n | strict leak % (95% CI) | loose leak % | arg leak % | bypass % | accuracy % | mean tool calls | mean retries |
|---|---|---|---|---|---|---|---|---|---|
| claude-haiku-5-5 | GATE | 900 | 0.2 (0.1 to 0.8) | 0.2 | 1.0 | 1.0 | 81.4 | 1.53 | 0.09 |
| claude-haiku-5-5 | OPTIONAL | 1200 | 50.0 (47.2 to 52.8) | 62.7 | 1.4 | 52.6 | 80.4 | 0.66 | 0.00 |
| claude-haiku-5-5 | REQUIRED | 1908 | 1.8 (1.3 to 2.5) | 15.8 | 2.2 | 0.9 | 73.9 | 1.39 | 0.00 |
| claude-opus-5-5 | GATE | 900 | 0.0 (0.0 to 0.4) | 0.0 | 0.7 | 0.1 | 81.6 | 1.41 | 0.06 |
| claude-opus-5-5 | OPTIONAL | 1200 | 30.9 (28.4 to 33.6) | 44.8 | 0.3 | 10.0 | 84.1 | 0.92 | 0.00 |
| claude-opus-5-5 | REQUIRED | 1500 | 2.7 (2.0 to 3.7) | 13.9 | 0.9 | 0.3 | 75.9 | 1.34 | 0.00 |
| claude-sonnet-5-5 | OPTIONAL | 900 | 45.3 (42.1 to 48.6) | 50.7 | 1.3 | 33.4 | 81.7 | 0.72 | 0.00 |
| claude-sonnet-5-5 | REQUIRED | 900 | 0.8 (0.4 to 1.6) | 10.0 | 1.1 | 0.4 | 81.3 | 1.34 | 0.00 |

## Sub-type shares among strict leaks, and wrong-source copies

| regime | pressure | n | leaked (strict) | L1 in-text % | L2 altered % | L3 unexplained % | L4 laundered % | wrong-source copies (final) |
|---|---|---|---|---|---|---|---|---|
| GATE | BREVITY | 600 | 1 | 0.0 | 0.0 | 0.0 | 100.0 | 0 |
| GATE | FAILURE | 600 | 1 | 0.0 | 0.0 | 0.0 | 100.0 | 0 |
| GATE | NEUTRAL | 600 | 0 | n/a | n/a | n/a | n/a | 0 |
| OPTIONAL | BREVITY | 900 | 249 | 80.7 | 0.0 | 17.3 | 2.0 | 0 |
| OPTIONAL | BREVITY_CTRL | 600 | 206 | 78.6 | 0.0 | 20.4 | 1.0 | 0 |
| OPTIONAL | FAILURE | 900 | 687 | 79.5 | 0.0 | 19.9 | 0.6 | 0 |
| OPTIONAL | NEUTRAL | 900 | 237 | 82.7 | 0.0 | 14.8 | 2.5 | 0 |
| REQUIRED | BREVITY | 900 | 6 | 16.7 | 0.0 | 0.0 | 83.3 | 0 |
| REQUIRED | BREVITY_CTRL | 600 | 3 | 33.3 | 0.0 | 0.0 | 66.7 | 0 |
| REQUIRED | FAILURE | 900 | 5 | 20.0 | 0.0 | 0.0 | 80.0 | 0 |
| REQUIRED | NEUTRAL | 1308 | 9 | 33.3 | 0.0 | 0.0 | 66.7 | 0 |
| REQUIRED | PERTURB | 600 | 60 | 6.7 | 75.0 | 0.0 | 18.3 | 0 |

## Paired tests across pressures (shared items, within regime and model)

| regime | model | pressure vs NEUTRAL | items paired | leaked_strict base % | leaked_strict pressure % | discordant (b, c) | McNemar exact p |
|---|---|---|---|---|---|---|---|
| GATE | claude-haiku-5-5 | BREVITY | 300 | 0.0 | 0.3 | (0, 1) | 1.000 |
| GATE | claude-haiku-5-5 | FAILURE | 300 | 0.0 | 0.3 | (0, 1) | 1.000 |
| GATE | claude-opus-5-5 | BREVITY | 300 | 0.0 | 0.0 | (0, 0) | 1.000 |
| GATE | claude-opus-5-5 | FAILURE | 300 | 0.0 | 0.0 | (0, 0) | 1.000 |
| OPTIONAL | claude-haiku-5-5 | BREVITY | 300 | 40.7 | 47.3 | (37, 57) | 0.049 |
| OPTIONAL | claude-haiku-5-5 | BREVITY_CTRL | 300 | 40.7 | 56.0 | (29, 75) | 0.000 |
| OPTIONAL | claude-haiku-5-5 | FAILURE | 300 | 40.7 | 56.0 | (21, 67) | 0.000 |
| OPTIONAL | claude-opus-5-5 | BREVITY | 300 | 4.0 | 13.7 | (4, 33) | 0.000 |
| OPTIONAL | claude-opus-5-5 | BREVITY_CTRL | 300 | 4.0 | 12.7 | (3, 29) | 0.000 |
| OPTIONAL | claude-opus-5-5 | FAILURE | 300 | 4.0 | 93.3 | (0, 268) | 0.000 |
| OPTIONAL | claude-sonnet-5-5 | BREVITY | 300 | 34.3 | 22.0 | (43, 6) | 0.000 |
| OPTIONAL | claude-sonnet-5-5 | FAILURE | 300 | 34.3 | 79.7 | (1, 137) | 0.000 |
| REQUIRED | claude-haiku-5-5 | BREVITY | 300 | 0.7 | 0.7 | (0, 0) | 1.000 |
| REQUIRED | claude-haiku-5-5 | BREVITY_CTRL | 300 | 0.7 | 0.7 | (0, 0) | 1.000 |
| REQUIRED | claude-haiku-5-5 | FAILURE | 300 | 0.7 | 0.7 | (0, 0) | 1.000 |
| REQUIRED | claude-opus-5-5 | BREVITY | 300 | 0.3 | 0.3 | (0, 0) | 1.000 |
| REQUIRED | claude-opus-5-5 | BREVITY_CTRL | 300 | 0.3 | 0.3 | (0, 0) | 1.000 |
| REQUIRED | claude-opus-5-5 | FAILURE | 300 | 0.3 | 0.3 | (0, 0) | 1.000 |
| REQUIRED | claude-sonnet-5-5 | BREVITY | 300 | 0.7 | 1.0 | (0, 1) | 1.000 |
| REQUIRED | claude-sonnet-5-5 | FAILURE | 300 | 0.7 | 0.7 | (0, 0) | 1.000 |

## Tool-perturbation probe

| regime | model | n (perturbed) | followed the tool % | hidden computation % | reported the discrepancy % | correct % |
|---|---|---|---|---|---|---|
| REQUIRED | claude-haiku-5-5 | 297 | 79.5 | 8.4 | 43.1 | 36.7 |
| REQUIRED | claude-opus-5-5 | 297 | 80.5 | 14.8 | 14.1 | 49.8 |

## After a calculator failure

| regime | model | n | reported the failure % | retried the tool (2+ calls) % | strict leak % | correct % |
|---|---|---|---|---|---|---|
| GATE | claude-haiku-5-5 | 300 | 1.0 | 99.0 | 0.3 | 81.0 |
| GATE | claude-opus-5-5 | 300 | 4.3 | 94.7 | 0.0 | 79.3 |
| OPTIONAL | claude-haiku-5-5 | 300 | 3.0 | 41.7 | 56.0 | 79.3 |
| OPTIONAL | claude-opus-5-5 | 300 | 10.7 | 0.3 | 93.3 | 84.0 |
| OPTIONAL | claude-sonnet-5-5 | 300 | 4.0 | 15.3 | 79.7 | 81.7 |
| REQUIRED | claude-haiku-5-5 | 300 | 3.3 | 99.0 | 0.7 | 80.3 |
| REQUIRED | claude-opus-5-5 | 300 | 9.3 | 94.3 | 0.3 | 79.7 |
| REQUIRED | claude-sonnet-5-5 | 300 | 43.7 | 98.0 | 0.7 | 80.7 |

## Does leakage predict errors?

Arm: original. Episodes 8808 over 300 items; leaked 1404 (wrong 23.9%), not leaked 7404 (wrong 17.3%); leaked-and-correct share 76.1%.
Mantel-Haenszel pooled relative risk of a wrong answer given strict leakage: 1.43 (bootstrap 95% interval 0.77 to 2.31, 1000 resamples; strata regime x pressure x K x bypass).

Arm: counterfactual. Episodes 3012 over 251 items; leaked 648 (wrong 25.9%), not leaked 2364 (wrong 19.2%); leaked-and-correct share 74.1%.
Mantel-Haenszel pooled relative risk of a wrong answer given strict leakage: 0.98 (bootstrap 95% interval 0.55 to 1.63, 1000 resamples; strata regime x pressure x K x bypass).
Tool calls per episode, counterfactual 1.12 versus original 1.14 (interpret the size of the recall effect only when these match).

## The price of the gate (and the contract)

| regime | pressure | model | n | retries / 100 | delivered after retry exhaustion % | accuracy of delivered % | rejected a correct reply % | rejections on prose only (answer line traceable) % | strict leak % (not a finding for GATE) | tool calls | output tokens | USD / 100 episodes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GATE | BREVITY | claude-haiku-5-5 | 300 | 1.3 | 0.3 | 81.3 | 0.7 | 0.7 | 0.3 | 1.03 | 168 | 0.05 |
| GATE | BREVITY | claude-opus-5-5 | 300 | 5.3 | 0.0 | 83.0 | 3.3 | 5.0 | 0.0 | 1.10 | 145 | 1.90 |
| GATE | FAILURE | claude-haiku-5-5 | 300 | 12.0 | 0.3 | 81.0 | 7.3 | 11.0 | 0.3 | 2.23 | 373 | 0.08 |
| GATE | FAILURE | claude-opus-5-5 | 300 | 5.3 | 0.0 | 79.3 | 3.7 | 5.0 | 0.0 | 2.01 | 259 | 2.85 |
| GATE | NEUTRAL | claude-haiku-5-5 | 300 | 14.0 | 0.0 | 82.0 | 9.0 | 13.0 | 0.0 | 1.32 | 280 | 0.06 |
| GATE | NEUTRAL | claude-opus-5-5 | 300 | 7.3 | 0.0 | 82.3 | 4.7 | 7.0 | 0.0 | 1.12 | 188 | 2.00 |
| REQUIRED | BREVITY | claude-haiku-5-5 | 300 | 0.0 | 0.0 | 80.7 | 0.0 | 0.0 | 0.7 | 1.06 | 176 | 0.05 |
| REQUIRED | BREVITY | claude-opus-5-5 | 300 | 0.0 | 0.0 | 83.7 | 0.0 | 0.0 | 0.3 | 1.03 | 141 | 1.78 |
| REQUIRED | BREVITY | claude-sonnet-5-5 | 300 | 0.0 | 0.0 | 81.0 | 0.0 | 0.0 | 1.0 | 1.02 | 116 | 0.86 |
| REQUIRED | BREVITY_CTRL | claude-haiku-5-5 | 300 | 0.0 | 0.0 | 80.7 | 0.0 | 0.0 | 0.7 | 1.10 | 219 | 0.05 |
| REQUIRED | BREVITY_CTRL | claude-opus-5-5 | 300 | 0.0 | 0.0 | 82.7 | 0.0 | 0.0 | 0.3 | 1.02 | 153 | 1.80 |
| REQUIRED | FAILURE | claude-haiku-5-5 | 300 | 0.0 | 0.0 | 80.3 | 0.0 | 0.0 | 0.7 | 2.09 | 336 | 0.07 |
| REQUIRED | FAILURE | claude-opus-5-5 | 300 | 0.0 | 0.0 | 79.7 | 0.0 | 0.0 | 0.3 | 1.95 | 251 | 2.72 |
| REQUIRED | FAILURE | claude-sonnet-5-5 | 300 | 0.0 | 0.0 | 80.7 | 0.0 | 0.0 | 0.7 | 1.98 | 241 | 1.36 |
| REQUIRED | NEUTRAL | claude-haiku-5-5 | 708 | 0.0 | 0.0 | 81.1 | 0.0 | 0.0 | 0.8 | 1.18 | 243 | 0.05 |
| REQUIRED | NEUTRAL | claude-opus-5-5 | 300 | 0.0 | 0.0 | 83.3 | 0.0 | 0.0 | 0.3 | 1.02 | 177 | 1.83 |
| REQUIRED | NEUTRAL | claude-sonnet-5-5 | 300 | 0.0 | 0.0 | 82.3 | 0.0 | 0.0 | 0.7 | 1.01 | 159 | 0.89 |

'rejected a correct reply' scores the rejected reply itself (its answer line against the gold) when the run or the replay recorded rejected_correct_n; otherwise it is the proxy 'a rejection occurred and the delivered answer is correct'.

## Noise floor (repeated seeds)

| cell | seeds | strict leak % per seed | range (points) |
|---|---|---|---|
| REQUIRED / NEUTRAL / claude-haiku-5-5 | 3 | 0.7, 1.0, 1.0 | 0.3 |

## Detector validation

(no mock rows: validation against scripted truth needs a dry run)

