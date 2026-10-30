#!/usr/bin/env python3
"""Figure and table data for the paper from a (re-labeled) results CSV.

  python3 -I figure_data.py all-relabeled.csv --out-dir ../results

Writes figure1-data.csv (strict and loose leakage with Wilson intervals by tier x regime x pressure, original items),
table1-grid.csv (the same plus bypass, argument leakage, accuracy, calls), table2-arms.csv (leakage versus error on the
original and counterfactual arms by tier and regime), and k-strata.csv (leakage by operation count).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze import load, wilson  # noqa: E402


def _i(r, k):
    v = r.get(k, "0")
    return int(v) if v not in ("", None) else 0


def grid(rows, out_dir):
    groups = defaultdict(list)
    for r in rows:
        if _i(r, "counterfactual") == 0:
            groups[(r["model"], r["regime"], r["pressure"])].append(r)
    with open(os.path.join(out_dir, "figure1-data.csv"), "w", newline="") as f1, open(os.path.join(out_dir, "table1-grid.csv"), "w", newline="") as f2:
        w1 = csv.writer(f1)
        w1.writerow(["tier", "regime", "pressure", "n", "strict_pct", "strict_lo", "strict_hi", "loose_pct", "loose_lo", "loose_hi"])
        w2 = csv.writer(f2)
        w2.writerow(["tier", "regime", "pressure", "n", "strict_pct", "strict_lo", "strict_hi", "loose_pct", "arg_leak_pct", "bypass_pct", "accuracy_pct", "mean_tool_calls", "retries_per_100", "reported_failure_pct"])
        for key in sorted(groups):
            g = groups[key]
            n = len(g)
            ps, lo, hi = wilson(sum(_i(r, "leaked_strict") for r in g), n)
            pl, llo, lhi = wilson(sum(_i(r, "leaked_loose") for r in g), n)
            w1.writerow([key[0], key[1], key[2], n, round(100 * ps, 1), round(100 * lo, 1), round(100 * hi, 1), round(100 * pl, 1), round(100 * llo, 1), round(100 * lhi, 1)])
            w2.writerow([key[0], key[1], key[2], n, round(100 * ps, 1), round(100 * lo, 1), round(100 * hi, 1), round(100 * pl, 1),
                         round(100 * sum(_i(r, "argument_leakage") for r in g) / n, 1), round(100 * sum(_i(r, "tool_bypass") for r in g) / n, 1),
                         round(100 * sum(_i(r, "correct") for r in g) / n, 1), round(sum(_i(r, "n_tool_calls") for r in g) / n, 2),
                         round(100 * sum(_i(r, "gate_rejections") for r in g) / n, 1), round(100 * sum(_i(r, "reported_failure") for r in g) / n, 1)])


def arms(rows, out_dir):
    groups = defaultdict(list)
    for r in rows:
        if r["pressure"] == "PERTURB" or r["regime"] not in ("OPTIONAL", "REQUIRED", "GATE"):
            continue
        groups[(r["model"], r["regime"], "counterfactual" if _i(r, "counterfactual") else "original")].append(r)
    with open(os.path.join(out_dir, "table2-arms.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tier", "regime", "arm", "n", "leaked_n", "leaked_wrong_pct", "traceable_n", "traceable_wrong_pct", "leaked_and_correct_pct", "accuracy_pct", "bypass_pct", "mean_tool_calls"])
        for key in sorted(groups):
            g = groups[key]
            le = [r for r in g if _i(r, "leaked_strict") == 1]
            nl = [r for r in g if _i(r, "leaked_strict") == 0]
            w.writerow([key[0], key[1], key[2], len(g), len(le),
                        round(100 * sum(1 for r in le if _i(r, "correct") == 0) / len(le), 1) if le else "",
                        len(nl), round(100 * sum(1 for r in nl if _i(r, "correct") == 0) / len(nl), 1) if nl else "",
                        round(100 * sum(1 for r in le if _i(r, "correct") == 1) / len(le), 1) if le else "",
                        round(100 * sum(_i(r, "correct") for r in g) / len(g), 1), round(100 * sum(_i(r, "tool_bypass") for r in g) / len(g), 1),
                        round(sum(_i(r, "n_tool_calls") for r in g) / len(g), 2)])


def strata(rows, out_dir):
    groups = defaultdict(list)
    for r in rows:
        if _i(r, "counterfactual") == 0 and r["pressure"] not in ("PERTURB",):
            groups[(r["model"], r["regime"], r["k_bucket"])].append(r)
    with open(os.path.join(out_dir, "k-strata.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tier", "regime", "k_bucket", "n", "strict_pct", "loose_pct", "accuracy_pct", "bypass_pct"])
        for key in sorted(groups):
            g = groups[key]
            n = len(g)
            w.writerow([key[0], key[1], key[2], n, round(100 * sum(_i(r, "leaked_strict") for r in g) / n, 1), round(100 * sum(_i(r, "leaked_loose") for r in g) / n, 1),
                        round(100 * sum(_i(r, "correct") for r in g) / n, 1), round(100 * sum(_i(r, "tool_bypass") for r in g) / n, 1)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rows = load(args.csv)
    grid(rows, args.out_dir)
    arms(rows, args.out_dir)
    strata(rows, args.out_dir)
    print(f"wrote figure1-data.csv, table1-grid.csv, table2-arms.csv, k-strata.csv to {args.out_dir} from {len(rows)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
