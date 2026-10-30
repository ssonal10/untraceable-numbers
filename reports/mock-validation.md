# Detector validation on the scripted mock (detector 1.0)

The scripted mock (`harness/clients.py`, `MockClient`) plants leaks with known ground truth: numbers computed in text, altered tool outputs and laundered tool arguments. It makes no API calls. This report validates the implementation against its own definition; it is not a human validation (see `results/labels-to-annotate.csv` for that sample).

Run with the released harness (detector 1.0), 2026-10-29:

```bash
cd harness
export P2_DATA_CACHE=$HOME/.cache/p2-leakage
python3 -I run_sweep.py --dry-run --datasets finqa,tatqa --n-per-bucket 10 --out ../results/mock-validation.csv
python3 -I analyze.py ../results/mock-validation.csv
```

Episodes: 2100 (finqa 1050, tatqa 1050), every regime and pressure cell of the mock grid.

Episode level, loose metric (any number in the reply), truth = the mock planted at least one text-computed, altered or laundered number:

| | flagged | not flagged |
|---|---|---|
| planted leak | TP 762 | FN 68 |
| no planted leak | FP 32 | TN 1238 |

Precision 96.0 percent, recall 91.8 percent.

By regime:

| regime | episodes | TP | FP | FN | TN |
|---|---|---|---|---|---|
| CONTRACT | 420 | 29 | 3 | 12 | 376 |
| GATE | 420 | 1 | 29 | 15 | 375 |
| NOTOOL | 420 | 395 | 0 | 25 | 0 |
| OPTIONAL | 420 | 178 | 0 | 10 | 232 |
| REQUIRED | 420 | 159 | 0 | 6 | 255 |

29 of the 32 false positives are GATE episodes. In each, the mock's reply was rejected and retried, the delivered answer matches a tainted tool output (L4), and the mock had cleared its record of planted leaks when the gate rejected the reply (`harness/clients.py`, the `gate_rejected` branch). The flag follows the definition, which closes over the whole episode trace; the mock's truth covers only the last scripted attempt. Outside GATE, precision is 761 of 764 (99.6 percent).

An earlier dry run (2,280 episodes including ConvFinQA, before the currency-prefix, scale and strict-metric fixes described in the paper, Section 7) gave 96.5 percent precision and 85.3 percent recall. The figures above replace it; the paper cites these.
