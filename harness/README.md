# Arithmetic-leakage harness

Measures how often a tool-equipped LLM still produces numbers that are not traceable to a tool output or to the input context, on public financial QA benchmarks (FinQA, TAT-QA, ConvFinQA), across tool regimes and pressure conditions. Standard-library Python 3.11+ for everything except the real API clients.

## Layout

| File | What |
|---|---|
| `calculator.py` | The tool: a restricted arithmetic evaluator (AST whitelist) with result ids (`r1`, `r2`), the `calculator` and `submit_answer` tool schemas, failure injection |
| `detector.py` | Number extraction, normalization (commas, currency, parentheses negatives, percent, scale words), matching with rounding tolerance, the explanation search, leak sub-types L1/L2/L3, argument leakage, tool bypass |
| `loaders.py` | FinQA, TAT-QA (arithmetic questions) and ConvFinQA (per turn, teacher-forced history) in their original repository formats; download into a cache; gold program parsing; stratified sampling by operation count |
| `clients.py` | `MockClient` (scripted policy with known ground truth), `AnthropicClient` (official SDK, lazy import, spending guard), `OpenAICompatClient` (open-weights endpoints; untested here) |
| `agent.py` | Conditions, prompts, the agent loop, the answer contract (`submit_answer` validation), the numbers gate with retries, DISTRACT padding, the results row |
| `run_sweep.py` | CLI; resumable CSV writer |
| `analyze.py` | Leakage tables, accuracy by leakage status, Mantel-Haenszel relative risk with a bootstrap interval, mitigation table, detector validation against the mock's truth |
| `cost_estimate.py` | Episode counts x measured tokens x list prices, with and without the Batches API discount |
| `selftest.py` | Zero-cost self-tests over every FinQA test and TAT-QA dev item (context numbers never flagged; gold-faithful replies never flagged; gold arguments never argument leaks; the scoring rule reproduces the datasets' answer strings; collision share) |
| `calibrate_l1.py` | L1 chance-match calibration: how often the explanation search explains a value that cannot be a computation over the context |
| `export_labels.py` | `export`: stratified labeling sample (120 flagged mentions + 40 unflagged episodes) with annotator columns; `kappa`: Cohen's kappa and detector precision/recall against the annotators |
| `run_parallel.py` | Runs cells as parallel `run_sweep.py` processes and merges the shards (the list-price synchronous alternative to `run_batches.py`) |
| `tests/test_harness.py` | Unit tests for the calculator, detector and program parser (`python3 -I tests/test_harness.py`) |
| `sample_data/mini_finqa.json` | Nine handmade items in FinQA format for offline dry runs and tests (synthetic numbers, no real filings) |

## Quick start

```bash
cd harness
python3 -I tests/test_harness.py                       # unit tests
python3 -I run_sweep.py --dry-run --mini --out mini.csv  # offline dry run, scripted mock
python3 -I analyze.py mini.csv

# dry run on the real public files (downloaded into P2_DATA_CACHE, default ~/.cache/p2-leakage)
P2_DATA_CACHE=/tmp/p2-cache python3 -I run_sweep.py --dry-run --datasets finqa,tatqa,convfinqa --n-per-bucket 4 --out dryrun.csv
python3 -I analyze.py dryrun.csv --markdown dryrun-analysis.md
python3 -I cost_estimate.py dryrun.csv --markdown cost.md
```

`-I` runs Python in isolated mode (no user site-packages, no PYTHONPATH); the scripts add their own directory to `sys.path`, so the downloaded data directory never has to be on the import path.

## Real runs

Install the SDK in a virtual environment (`uv venv && uv pip install anthropic`), authenticate (`ANTHROPIC_API_KEY`, or `ant auth login`), then:

```bash
P2_MAX_CALLS=40 python3 run_sweep.py --client anthropic --model claude-haiku-5-5 \
  --datasets finqa --n-per-bucket 2 --cells REQUIRED:NEUTRAL,REQUIRED:BREVITY --out pilot.csv
```

`P2_MAX_CALLS` (default 50) is a hard stop per process; raise it deliberately for the pilot and the sweep. Model ids come from `--model` or `P2_MODEL` (default `claude-opus-5-5`); the size ladder in the design is `claude-haiku-5-5`, `claude-sonnet-5-5`, `claude-opus-5-5`. Effort (`--efforts low,high`) replaces temperature on current Claude models, which reject non-default sampling parameters; `--temperatures` is only for models that accept them (open-weights route: `--client openai-compat`, `P2_OPENAI_BASE_URL`, `P2_OPENAI_API_KEY`, `P2_OPENAI_MODEL`).

Refusals are recorded as `stop_reason = refusal`; no fallback model is configured so the model factor stays clean.

## Conditions

Regimes: `NOTOOL`, `OPTIONAL`, `REQUIRED`, `CONTRACT`, `GATE`. Pressures: `NEUTRAL`, `BREVITY`, `BUDGET`, `FAILURE`, `DISTRACT`. Formats: `FREE`, `JSON`. See the paper for definitions; prompts are in `agent.py` and printed verbatim in the paper (Table I).

## Results CSV columns

Columns of `results/results-final-relabeled.csv` (69): `dataset`, `item_id`, `k`, `k_bucket`, `regime`, `pressure`, `effort`, `fmt`, `temperature`, `model`, `seed`, `turns`, `n_tool_calls`, `n_tool_errors`, `gate_rejections`, `rejected_leaks`, `rejected_correct_n`, `rejected_strict_clean_n`, `contract_violations`, `contract_forced`, `submitted_via_contract`, `stop_reason`, `model_reported`, `api_error`, `perturbed`, `perturbed_true_value`, `pred`, `gold`, `correct`, `followed_tool`, `hidden_computation`, `reported_discrepancy`, `reported_failure`, `counterfactual`, `input_tokens`, `output_tokens`, `cache_read_input_tokens`, `prompt_hash`, `elapsed_s`, `ts`, `leaked_strict`, `leaked_loose`, `n_final_numbers`, `n_final_leaked`, `n_text_numbers`, `n_text_leaked`, `argument_leakage`, `n_arg_leaks`, `tool_bypass`, `excluded_count`, `L1_final`, `L2_final`, `L3_final`, `L1_loose`, `L2_loose`, `L3_loose`, `L4_final`, `L4_loose`, `wrong_source_final`, `leaked_but_gold_correct`, `truth_text_computed_n`, `truth_emitted_leaks_n`, `truth_altered`, `truth_laundered`, `truth_bypass`, `truth_fallback`, `final_text`, `mentions`, `replay_diverged`.

Labels in `mentions`: COPY_CTX, TOOL_OUT, TOOL_IN_ECHO, CONST, L1 (in-text computation), L2 (altered tool output), L3 (unexplained), L4 (laundered: tool output fed by an untraceable argument), WRONG_SRC (copied from a distractor document), EXCLUDED. Precedence among leak types: L4, L2, L1, L3.

## Known limits (to state in the paper)

- A model that computes in hidden reasoning and then calls the tool with the result shows up only as argument leakage when the argument is untraceable.
- A computed number that happens to equal a context number is labeled a copy (a detector miss by construction); the human-labeled sample estimates how often this happens.
- The scoring rule is the Program-of-Thoughts finqa_equal rule (math.isclose, rel_tol 0.001, with the x100 and /100 alternates) extended by rounding to the prediction's printed decimals, because the datasets' own answer strings are rounded (FinQA writes "14%" for an executed 0.14464). selftest.py reports how many answer strings each rule reproduces; the public answer string disagrees with the executed gold program in the remaining FinQA cases (for example "704.25" against an executed 705.25); these were not adjudicated, and accuracy is scored against the program value.
- `OpenAICompatClient` has not been exercised against a live endpoint in this study.
- Batch-mode (wave) execution through the Message Batches API is in `run_batches.py`; `run_parallel.py` is the list-price synchronous alternative.
- Sign conventions: "(197)" in a table is -197 and also matches a restated 197 (copy, not computation); a minus glued to a preceding number ("2,303-1,780") or spaced on both sides is a binary operator; "2012 -4" is a sign; tool arguments match without regard to sign.
