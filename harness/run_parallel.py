#!/usr/bin/env python3
"""Run cells of the sweep as parallel processes (one run_sweep.py per cell), then merge the shards.

  python3 -I run_parallel.py --workers 4 --out results.csv -- --client anthropic --model claude-haiku-5-5 \
      --datasets finqa,tatqa --n-per-bucket 50 --cells REQUIRED:NEUTRAL,REQUIRED:BREVITY,REQUIRED:FAILURE

Everything after `--` is passed to run_sweep.py unchanged, except --cells (split one cell per process) and --out
(each process writes <out>.<REGIME>.<PRESSURE>.csv; the shards are merged into <out> at the end; re-running merges
again and skips nothing inside a shard, because run_sweep.py itself is resumable per shard). Traces go to one JSONL
per shard when --traces is given. The P2_MAX_CALLS guard applies per process: set it to the per-cell budget.

This is the list-price alternative to the Batches API wave mode (not implemented); it only shortens wall-clock time.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("sweep_args", nargs=argparse.REMAINDER, help="arguments for run_sweep.py after --")
    args = ap.parse_args()
    rest = [a for a in args.sweep_args if a != "--"]
    cells = "all"
    traces = None
    cleaned: list[str] = []
    i = 0
    while i < len(rest):
        if rest[i] == "--cells":
            cells = rest[i + 1]
            i += 2
        elif rest[i] == "--out":
            i += 2
        elif rest[i] == "--traces":
            traces = rest[i + 1]
            i += 2
        else:
            cleaned.append(rest[i])
            i += 1
    if cells == "all":
        from agent import PRESSURES, REGIMES  # noqa: E402

        cell_list = [f"{r}:{p}" for r in REGIMES for p in PRESSURES]
    else:
        cell_list = [c.strip() for c in cells.split(",") if c.strip()]

    def run_cell(cell: str) -> tuple[str, int]:
        shard = f"{args.out}.{cell.replace(':', '.')}.csv"
        cmd = [sys.executable, "-I", os.path.join(HERE, "run_sweep.py"), *cleaned, "--cells", cell, "--out", shard]
        if traces:
            cmd += ["--traces", f"{traces}.{cell.replace(':', '.')}.jsonl"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        tail = (proc.stdout or "").strip().splitlines()[-1:] or [""]
        print(f"[{cell}] exit {proc.returncode}: {tail[0]}" + (f" | stderr: {proc.stderr.strip()[-200:]}" if proc.returncode else ""))
        return shard, proc.returncode

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(run_cell, cell_list))
    # merge shards
    header: list[str] | None = None
    n = 0
    with open(args.out, "w", newline="") as out:
        writer = None
        for shard, _ in results:
            if not os.path.exists(shard):
                continue
            with open(shard) as fh:
                for row in csv.DictReader(fh):
                    if writer is None:
                        header = list(row.keys())
                        writer = csv.DictWriter(out, fieldnames=header)
                        writer.writeheader()
                    writer.writerow({k: row.get(k, "") for k in header})
                    n += 1
    print(f"merged {n} rows from {len(results)} shards into {args.out}")
    return 0 if all(rc == 0 for _, rc in results) else 1


if __name__ == "__main__":
    sys.exit(main())
