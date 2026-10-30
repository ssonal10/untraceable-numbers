#!/usr/bin/env python3
"""Cost estimate for the sweep from measured token counts and list prices.

  python3 -I cost_estimate.py results.csv [--thinking-per-turn 300] [--markdown cost.md]

Token counts per turn come from the dry-run CSV (the mock approximates input tokens as
characters / 4 of the full request, which tracks real counts within about 20 percent for
English text and Markdown tables; output tokens are a placeholder and are replaced by
--output-per-turn plus --thinking-per-turn). Prices are per million tokens, list prices
cached from the API reference on 2026-10-06; re-check before spending.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PRICES = {  # input, output per million tokens
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}
BATCH_DISCOUNT = 0.5

H, S, O = "claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"
GRIDS = {
    # name: {model: episodes}
    "pilot: R2 x {P0,P1,P3} x 30 items, plus one cell repeated 3x (noise floor)": {H: 90 + 90},
    "minimum design: R1,R2,R4 x P0,P1,P3 x 300 items (150 FinQA + 150 TAT-QA) x small and large; + counterfactual arm (R2 x 3 x 300) + PERTURB probe (R2 x 300) on both": {H: 2700 + 900 + 300, O: 2700 + 900 + 300},
    "reduced design: R0..R4 x P0,P1,P3 x 150 items x 3 sizes; + effort/format arms on the mid model": {H: 2250, S: 2250 + 600, O: 2250},
    "core: R0..R4 x P0..P4 x 300 items x 3 sizes; + effort/format arms (1,200 per model); + counterfactual and PERTURB arms (1,200 per model)": {H: 7500 + 2400, S: 7500 + 2400, O: 7500 + 2400},
}


def measure(path: str) -> dict[str, float]:
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    by_regime: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_regime[r["regime"]].append(r)
    turns = {k: sum(int(r["turns"]) for r in v) / len(v) for k, v in by_regime.items()}
    inp_per_turn = sum(int(r["input_tokens"]) for r in rows) / max(1, sum(int(r["turns"]) for r in rows))
    mean_turns = sum(int(r["turns"]) for r in rows) / len(rows)
    return {"input_per_turn": inp_per_turn, "mean_turns": mean_turns, "turns_by_regime": turns, "n": len(rows)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--input-per-turn", type=float, default=None, help="override measured mean input tokens per turn")
    ap.add_argument("--output-per-turn", type=float, default=180.0, help="visible output tokens per turn (text + tool call)")
    ap.add_argument("--thinking-per-turn", type=float, default=300.0, help="thinking tokens per turn at low effort")
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args()
    m = measure(args.csv)
    inp = args.input_per_turn or m["input_per_turn"]
    turns = m["mean_turns"]
    out_tok = args.output_per_turn + args.thinking_per_turn
    lines = [
        "# Cost estimate (list prices cached 2026-10-06; re-check before spending)",
        "",
        f"Measured on {m['n']} dry-run episodes: mean input tokens per turn {inp:,.0f} (whole request, no caching), mean turns per episode {turns:.2f}",
        f"(by regime: {', '.join(f'{k} {v:.2f}' for k, v in sorted(m['turns_by_regime'].items()))}). Assumed output per turn: {args.output_per_turn:.0f} visible + {args.thinking_per_turn:.0f} thinking = {out_tok:.0f} tokens.",
        "",
        "| grid | model | episodes | input tokens (M) | output tokens (M) | cost USD | cost with Batches API (50%) |",
        "|---|---|---|---|---|---|---|",
    ]
    grand = {}
    for name, per_model in GRIDS.items():
        total = total_b = 0.0
        for model, episodes in per_model.items():
            in_tok = episodes * turns * inp
            out_tk = episodes * turns * out_tok
            pi, po = PRICES[model]
            cost = in_tok / 1e6 * pi + out_tk / 1e6 * po
            total += cost
            total_b += cost * BATCH_DISCOUNT
            lines.append(f"| {name} | {model} | {episodes:,} | {in_tok / 1e6:.1f} | {out_tk / 1e6:.1f} | {cost:,.0f} | {cost * BATCH_DISCOUNT:,.0f} |")
        lines.append(f"| {name} | **total** | | | | **{total:,.0f}** | **{total_b:,.0f}** |")
        grand[name] = (total, total_b)
    lines += ["", "Notes:", "- Input tokens count the whole request on every turn; prompt caching of the item context would cut the input side of multi-turn episodes (cache reads are priced at a fraction of input) once the prefix passes the model's minimum cacheable size.",
              "- Output tokens include thinking, which cannot be disabled on claude-opus-5-5; `high` effort arms cost more per turn.",
              "- Open-weights models are not priced here; hosted inference for a mid-size model is typically a fraction of the Sonnet row.",
              "- The pilot's purpose is to replace these assumptions with measured usage from the API's usage fields."]
    text = "\n".join(lines)
    print(text)
    if args.markdown:
        with open(args.markdown, "w") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
