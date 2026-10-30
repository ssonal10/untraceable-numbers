# Untraceable Numbers

Code, results and traces for the paper "Untraceable Numbers: Verifying the Provenance of Computed Values in Tool-Using LLM Agents" (Shipra Sonal, 2026).

An *arithmetic leak* is a number in an agent's reply that is neither a copy of the input nor traceable, through the tool trace, to a calculator result whose own inputs were traceable. This repository holds the detector, the agent loop and batch runner used in the study, every episode trace, the labeled results and the scripts that regenerate the paper's tables from the traces without any API call.

## Contents

| Folder | What |
|---|---|
| `harness/` | Calculator tool, detector, agent loop, prompts (`agent.py`), Message Batches runner, replay, analysis, self-tests, L1 calibration, sensitivity, labeling export. Standard-library Python 3.11+ except the API client. See `harness/README.md`. |
| `results/` | `results-final-relabeled.csv` (12,420 grid and arm episodes, one row each, relabeled with the final detector), `pilot-relabeled.csv` (126 pilot episodes), table and figure data, `all-analysis.md`, `labels-to-annotate.csv` (stratified sample for human validation, annotator columns empty), `cost-log.csv` (one row per batch with batch id, tokens and price), `mock-validation.csv` (scripted-mock episodes with planted leaks), `counterfactual-skips.csv` (the 49 items the counterfactual arm skipped, with reasons). |
| `traces/` | Full episode traces as gzipped JSONL (every turn, tool call and returned value), plus the batch-runner logs (local paths replaced by `<run-dir>`). |
| `reports/` | Detector self-tests, L1 chance-match calibration (FinQA, TAT-QA), the detector sensitivity table, the scripted-mock validation (`mock-validation.md`) and exclusion counts with counterfactual skip reasons (`exclusions-and-cf-skips.md`). |

## Detector version

Every released label was produced by detector version 1.0 (`DETECTOR_VERSION` in `harness/detector.py`), the version tagged `v1.0` in this repository. Labels are a function of the trace and the detector version; rerun `replay.py` to relabel the traces under a later version.

## Data

Public benchmarks only, downloaded from their original repositories by `harness/loaders.py`:

- FinQA (test split): https://github.com/czyssrs/FinQA (MIT license). Chen et al., EMNLP 2021.
- TAT-QA (dev split, arithmetic questions): https://github.com/NExTplusplus/TAT-QA (data under CC BY 4.0). Zhu et al., ACL-IJCNLP 2021.

Sample: 150 items per dataset, stratified by gold operation count K (50 each for K = 1, 2, 3+), fixed seed (see `loaders.py` and `run_batches.py`). The traces contain excerpts of these public datasets as model input.

## Models

claude-haiku-5-5 (small), claude-sonnet-5-5 (mid; OPTIONAL and REQUIRED rows only) and claude-opus-5-5 (large), effort low, at most 4,096 output tokens, Anthropic Message Batches API, SDK 2.1.0, one run per cell. Runs on 2026-10-21 and 2026-10-24; 12,420 episodes plus a 126-episode pilot; 61.76 USD at batch prices over 63 batches (`results/cost-log.csv`). Temperature is not set (these models reject the parameter), so the API default applies; the `seed` column is a replicate index, not an API parameter. USD figures in `all-analysis.md` are at batch prices, half the list price.

## Reproduce

No API key is needed for steps 1 to 3.

```bash
cd harness
python3 -I tests/test_harness.py                        # unit tests (calculator, detector, program parser)

# 1. Tables from the released results
python3 -I analyze.py ../results/results-final-relabeled.csv --markdown analysis.md

# 2. Relabel every episode from the traces with the current detector (downloads FinQA and TAT-QA into the cache)
gunzip -k ../traces/*.jsonl.gz                          # the scripts read plain JSONL
export P2_DATA_CACHE=$HOME/.cache/p2-leakage
python3 -I replay.py ../traces/grid-haiku.jsonl ../traces/grid-opus.jsonl ../traces/grid-sonnet.jsonl \
    ../traces/cf-haiku.jsonl ../traces/cf-opus.jsonl --out relabeled.csv
python3 -I figure_data.py relabeled.csv --out-dir ../results

# 3. Self-tests, chance-match calibration and detector sensitivity (zero model cost)
python3 -I selftest.py --markdown selftest-report.md
python3 -I calibrate_l1.py --dataset finqa --n 150 --markdown l1-chance-match.md   # and --dataset tatqa
python3 -I sensitivity.py ../traces/grid-*.jsonl --out-dir sens --markdown sensitivity.md
python3 -I run_sweep.py --dry-run --datasets finqa,tatqa --n-per-bucket 10 --out mock.csv   # scripted mock, no API
python3 -I analyze.py mock.csv                                 # ends with the detector validation

# 4. New runs (costs money): put an Anthropic API key in ~/.config/untraceable-numbers/anthropic_key
python3 run_batches.py --backend anthropic --model claude-haiku-5-5 --datasets finqa,tatqa --n-per-bucket 50 \
    --cells OPTIONAL:NEUTRAL,REQUIRED:NEUTRAL --out run.csv --traces run.jsonl \
    --cost-log ../results/cost-log.csv --cap-usd 5 --tag my-run
```

Check each script's `--help` for the exact options. Seventeen of the 900 small-tier GATE episodes are marked `replay_diverged`: their live gate rejections used an earlier extractor version (see the paper, Section 7).

## AI assistance

The harness, analysis and plot code were written with Claude (Anthropic) as a coding assistant, from the author's design; the author reviewed the code and verified the results. The Claude models evaluated in the study are objects of measurement, separate from this use.

## License

MIT (see `LICENSE`) for the code and the results files. The datasets keep their own licenses (above).

## Citation

S. Sonal, "Untraceable Numbers: Verifying the Provenance of Computed Values in Tool-Using LLM Agents," 2026 (under review). Code and data: tag `v1.0` of this repository.
