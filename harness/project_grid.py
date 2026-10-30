#!/usr/bin/env python3
"""Project the cost of grid components from measured pilot usage (per model: tokens per turn, turns per episode by regime).

  python3 -I project_grid.py pilot-haiku.csv pilot-opus.csv --cost-log ../results/cost-log.csv --cap-usd 80 \
      --plan "claude-haiku-5-5:OPTIONAL,REQUIRED,GATE:NEUTRAL,BREVITY,FAILURE:300" \
      --plan "claude-opus-5-5:REQUIRED:NEUTRAL,BREVITY,FAILURE:300" ...

Each --plan is model:regimes:pressures:items[:repeats]. Prices are list prices times the batch factor. Usage is taken from
the pilot rows of the same model (and the same regime where available, else the model's mean), so that thinking tokens
and turn counts are measured rather than assumed.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_batches import BATCH_FACTOR, PRICES, ledger_total  # noqa: E402


def measure(paths: list[str]) -> dict[str, dict]:
    rows = []
    for p in paths:
        with open(p) as fh:
            rows.extend(csv.DictReader(fh))
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("api_error"):
            continue
        by_model[r["model"]].append(r)
    out: dict[str, dict] = {}
    for m, rs in by_model.items():
        turns = sum(int(r["turns"]) for r in rs)
        inp = sum(int(r["input_tokens"]) for r in rs) / max(1, turns)
        outp = sum(int(r["output_tokens"]) for r in rs) / max(1, turns)
        by_regime = defaultdict(list)
        for r in rs:
            by_regime[r["regime"]].append(int(r["turns"]))
        out[m] = {"n": len(rs), "input_per_turn": inp, "output_per_turn": outp, "turns": turns / len(rs),
                  "turns_by_regime": {k: sum(v) / len(v) for k, v in by_regime.items()}}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--plan", action="append", default=[])
    ap.add_argument("--cost-log", default=None)
    ap.add_argument("--cap-usd", type=float, default=80.0)
    ap.add_argument("--regime-turn-fallback", type=float, default=None)
    args = ap.parse_args()
    meas = measure(args.csvs)
    print("Measured from the pilot (batch results):")
    for m, d in meas.items():
        print(f"  {m}: {d['n']} episodes, {d['input_per_turn']:,.0f} input tokens/turn, {d['output_per_turn']:,.0f} output tokens/turn (incl. thinking), {d['turns']:.2f} turns/episode; by regime {', '.join(f'{k} {v:.2f}' for k, v in d['turns_by_regime'].items())}")
    spent = ledger_total(__import__("pathlib").Path(args.cost_log)) if args.cost_log else 0.0
    print(f"\nSpent so far: {spent:.2f} USD; cap {args.cap_usd:.2f} USD\n")
    print("| plan | model | episodes | est. turns | input tokens (M) | output tokens (M) | cost USD (batch) |")
    print("|---|---|---|---|---|---|---|")
    total = 0.0
    for spec in args.plan:
        parts = spec.split(":")
        model, regimes, pressures, items = parts[0], parts[1].split(","), parts[2].split(","), int(parts[3])
        repeats = int(parts[4]) if len(parts) > 4 else 1
        d = meas.get(model)
        if d is None:
            print(f"| {spec} | {model} | (no pilot rows for this model) | | | | |")
            continue
        pi, po = PRICES[model]
        episodes = len(regimes) * len(pressures) * items * repeats
        turns = 0.0
        for rg in regimes:
            turns += d["turns_by_regime"].get(rg, args.regime_turn_fallback or d["turns"]) * len(pressures) * items * repeats
        in_tok = turns * d["input_per_turn"]
        out_tok = turns * d["output_per_turn"]
        cost = BATCH_FACTOR * (in_tok * pi + out_tok * po) / 1e6
        total += cost
        print(f"| {spec} | {model} | {episodes:,} | {turns:,.0f} | {in_tok / 1e6:.2f} | {out_tok / 1e6:.2f} | {cost:.2f} |")
    print(f"| **total** | | | | | | **{total:.2f}** |")
    print(f"\nProjected cumulative: {spent + total:.2f} USD of the {args.cap_usd:.2f} USD cap ({'FITS' if spent + total <= args.cap_usd else 'OVER'}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
