#!/usr/bin/env python3
"""Run the condition sweep and write one CSV row per episode.

Examples
  # dry run with the scripted mock on the bundled mini sample (no download, no API):
  python3 -I run_sweep.py --dry-run --mini --out results.csv

  # dry run on the real public files (downloaded into the cache on first use):
  P2_DATA_CACHE=/path/to/cache python3 -I run_sweep.py --dry-run --datasets finqa,tatqa,convfinqa \
      --n-per-bucket 5 --out results.csv

  # real pilot (needs ANTHROPIC_API_KEY or an `ant auth login` profile; refuses beyond P2_MAX_CALLS):
  P2_MAX_CALLS=40 python3 -I run_sweep.py --client anthropic --model claude-haiku-5-5 \
      --datasets finqa --n-per-bucket 2 --cells REQUIRED:NEUTRAL,REQUIRED:BREVITY --out pilot.csv

The run is resumable: rows already in --out are skipped.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # keep imports working under python -I

from agent import PRESSURES, REGIMES, Condition, RunConfig, run_episode  # noqa: E402
from clients import BudgetExceeded, MockPolicy, make_client  # noqa: E402
from loaders import bucket_counts, counterfactual, load_dataset, load_mini_sample, stratified_sample  # noqa: E402

HERE = Path(__file__).resolve().parent


def parse_cells(spec: str) -> list[tuple[str, str]]:
    if spec in ("all", ""):
        return [(r, p) for r in REGIMES for p in PRESSURES]
    cells = []
    for part in spec.split(","):
        r, p = part.strip().split(":")
        if r not in REGIMES or p not in PRESSURES:
            raise SystemExit(f"unknown cell {part!r}; regimes {REGIMES}, pressures {PRESSURES}")
        cells.append((r, p))
    return cells


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--client", default="mock", choices=["mock", "anthropic", "openai-compat"])
    ap.add_argument("--dry-run", action="store_true", help="alias for --client mock")
    ap.add_argument("--model", default=None, help="model id for real clients (default from P2_MODEL)")
    ap.add_argument("--datasets", default="finqa", help="comma list of finqa,tatqa,convfinqa")
    ap.add_argument("--mini", action="store_true", help="use the bundled miniature sample instead of downloads")
    ap.add_argument("--n-per-bucket", type=int, default=3, help="items per operation-count bucket per dataset")
    ap.add_argument("--buckets", default="1,2,3+")
    ap.add_argument("--cells", default="all", help="REGIME:PRESSURE pairs, comma separated, or 'all'")
    ap.add_argument("--efforts", default="low")
    ap.add_argument("--fmts", default="FREE")
    ap.add_argument("--temperatures", default="", help="comma list; only for models that accept temperature")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1, help="seeds per cell (for variance)")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--limit-items", type=int, default=None, help="cap items per dataset before sampling (speed)")
    ap.add_argument("--mock-leak-p", type=float, default=None)
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--counterfactual", action="store_true", help="replace sampled items by counterfactual-value versions (perturbed context and gold)")
    ap.add_argument("--traces", default=None, help="JSONL path for full episode records (messages, tool calls, usage, mentions)")
    args = ap.parse_args()

    kind = "mock" if args.dry_run else args.client
    policy = MockPolicy(seed=args.seed)
    if args.mock_leak_p is not None:
        policy.leak_p = args.mock_leak_p
    client = make_client(kind, model=args.model, policy=policy)

    buckets = tuple(b.strip() for b in args.buckets.split(","))
    items = []
    distractor_pool: dict[str, list] = {}
    if args.mini:
        pool = load_mini_sample(HERE / "sample_data" / "mini_finqa.json")
        items.extend(stratified_sample(pool, args.n_per_bucket, args.seed, buckets))
        distractor_pool["finqa"] = pool
        print(f"mini sample: {len(pool)} items, buckets {bucket_counts(pool)}")
    else:
        for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
            pool = load_dataset(ds, limit=args.limit_items)
            chosen = stratified_sample(pool, args.n_per_bucket, args.seed, buckets)
            print(f"{ds}: loaded {len(pool)} items, buckets {bucket_counts(pool)}; sampled {len(chosen)}")
            items.extend(chosen)
            distractor_pool[ds] = pool

    if args.counterfactual:
        cf_items = []
        for it in items:
            if it.history:  # ConvFinQA later turns depend on earlier gold answers; counterfactual only for first turns
                continue
            cf = counterfactual(it, args.seed)
            if cf is not None:
                cf_items.append(cf)
        print(f"counterfactual: {len(cf_items)} of {len(items)} items perturbed (others skipped: literal not found in context)")
        items = cf_items

    cells = parse_cells(args.cells)
    efforts = [e.strip() for e in args.efforts.split(",") if e.strip()]
    fmts = [f.strip() for f in args.fmts.split(",") if f.strip()]
    temps: list[float | None] = [float(t) for t in args.temperatures.split(",") if t.strip()] or [None]
    conds = [Condition(r, p, e, f, t) for (r, p) in cells for e in efforts for f in fmts for t in temps]

    out = Path(args.out)
    done: set[tuple[str, str, str, str]] = set()
    if out.exists():
        with out.open() as fh:
            for row in csv.DictReader(fh):
                done.add((row["item_id"], f"{row['regime']}|{row['pressure']}|{row['effort']}|{row['fmt']}|{row['temperature']}", row["model"], row["seed"]))
    cfg = RunConfig()
    rng = random.Random(args.seed)
    n_new = 0
    writer = None
    fh = out.open("a", newline="")
    trace_fh = open(args.traces, "a") if args.traces else None
    try:
        for cond in conds:
            for item in items:
                for rep in range(args.repeats):
                    seed = args.seed + rep
                    key = (item.id, f"{cond.regime}|{cond.pressure}|{cond.effort}|{cond.fmt}|{'' if cond.temperature is None else cond.temperature}", getattr(client, "model_name", "?"), str(seed))
                    if key in done:
                        continue
                    if args.max_episodes is not None and n_new >= args.max_episodes:
                        raise StopIteration
                    pool = [d for d in distractor_pool.get(item.dataset, []) if d.id != item.id and d.context_text != item.context_text]
                    distractors = rng.sample(pool, min(cfg.distract_tables, len(pool))) if cond.pressure == "DISTRACT" else []
                    try:
                        row = run_episode(client, item, cond, cfg, seed=seed, distractors=distractors, trace_sink=trace_fh)
                    except BudgetExceeded as e:
                        print(f"stopped: {e}")
                        raise StopIteration
                    if writer is None:
                        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
                        if out.stat().st_size == 0:
                            writer.writeheader()
                    writer.writerow(row)
                    fh.flush()
                    n_new += 1
                    if n_new % 50 == 0:
                        print(f"  {n_new} episodes written")
    except StopIteration:
        pass
    finally:
        fh.close()
        if trace_fh is not None:
            trace_fh.close()
    print(f"wrote {n_new} new episodes to {out} (skipped {len(done)} existing); client calls: {getattr(client, 'calls', '?')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
