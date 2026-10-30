# Exclusion counts and counterfactual skip reasons (detector 1.0)

## Excluded numbers

The detector excludes years present in the context and integers from 0 to 12 in prose (list markers, step numbers) before labeling. The column `excluded_count` in `results/results-final-relabeled.csv` gives the count per episode. Totals:

| arm | model | regime | episodes | excluded numbers | episodes with at least one exclusion |
|---|---|---|---|---|---|
| counterfactual | claude-haiku-5-5 | OPTIONAL | 753 | 2200 | 431 |
| counterfactual | claude-haiku-5-5 | REQUIRED | 753 | 2224 | 457 |
| counterfactual | claude-opus-5-5 | OPTIONAL | 753 | 3684 | 616 |
| counterfactual | claude-opus-5-5 | REQUIRED | 753 | 3719 | 638 |
| original | claude-haiku-5-5 | GATE | 900 | 2469 | 524 |
| original | claude-haiku-5-5 | OPTIONAL | 1200 | 3641 | 735 |
| original | claude-haiku-5-5 | REQUIRED | 1608 | 5533 | 1158 |
| original | claude-opus-5-5 | GATE | 900 | 4019 | 772 |
| original | claude-opus-5-5 | OPTIONAL | 1200 | 5516 | 997 |
| original | claude-opus-5-5 | REQUIRED | 1200 | 5328 | 1029 |
| original | claude-sonnet-5-5 | OPTIONAL | 900 | 3409 | 627 |
| original | claude-sonnet-5-5 | REQUIRED | 900 | 3362 | 660 |
| probe | claude-haiku-5-5 | REQUIRED | 300 | 1250 | 253 |
| probe | claude-opus-5-5 | REQUIRED | 300 | 1553 | 267 |
| all | | | 12420 | 47907 | 9164 |

Switching the exclusions off leaves strict and loose leakage unchanged on the original items (`reports/sensitivity-v2.md`).

## Counterfactual skips

The counterfactual arm perturbs the numbers the gold program reads, in the context and in the program (`harness/loaders.py`, `counterfactual`). An item is skipped when this cannot be done. 251 of the 300 sampled items were perturbed; 49 were skipped:

| dataset | reason | items |
|---|---|---|
| finqa | a literal the gold program reads is not written verbatim in the context | 8 |
| tatqa | a literal the gold program reads is not written verbatim in the context | 39 |
| tatqa | a table row read by a table operation was not found | 2 |

Per-item ids and reasons: `results/counterfactual-skips.csv` (recomputed from the public files with seed 0; the set of skipped ids equals the 300 sampled items minus the 251 counterfactual items in the results file).
