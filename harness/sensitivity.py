#!/usr/bin/env python3
"""Detector sensitivity table (Appendix B): re-label the stored traces under alternative detector settings and report
strict and loose leakage by regime for each setting. No API calls.

  P2_DATA_CACHE=... python3 -I sensitivity.py traces1.jsonl traces2.jsonl ... --out-dir /tmp/sens --markdown sensitivity.md
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

SETTINGS = {
    "default": {},
    "rel_tol 0.005": {"rel_tol": 0.005},
    "rel_tol 0.01": {"rel_tol": 0.01},
    "no absolute tolerance": {"abs_tol": 0.0},
    "sign sensitive": {"sign_insensitive": False},
    "scale re-expression against context on": {"allow_rescale_match": True},
    "percent re-expression of tool outputs off": {"rescale_tool_outputs": False},
    "no provenance closure (L4 off)": {"provenance_closure": False},
    "exclusions off": {"exclude_years": False, "exclude_list_markers": False},
}


def summarize(path: str) -> dict[tuple, dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r.get("counterfactual", "0") == "1" or r["pressure"] == "PERTURB":
                continue
            groups[(r["model"], r["regime"])].append(r)
    out = {}
    for k, g in groups.items():
        n = len(g)
        out[k] = {"n": n, "strict": sum(int(r["leaked_strict"]) for r in g) / n, "loose": sum(int(r["leaked_loose"]) for r in g) / n}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    results: dict[str, dict] = {}
    for name, cfg in SETTINGS.items():
        out = os.path.join(args.out_dir, name.replace(" ", "_").replace("(", "").replace(")", "") + ".csv")
        cmd = [sys.executable, "-I", os.path.join(HERE, "replay.py"), *args.traces, "--out", out, "--config", json.dumps(cfg)]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        results[name] = summarize(out)
    keys = sorted({k for v in results.values() for k in v})
    lines = ["# Detector sensitivity (strict / loose episode leakage in percent, by model and regime; original items, probe excluded)", "",
             "| setting | " + " | ".join(f"{m.split('-')[1]} {reg}" for m, reg in keys) + " |", "|---|" + "---|" * len(keys)]
    for name, res in results.items():
        cells = []
        for k in keys:
            d = res.get(k)
            cells.append(f"{100 * d['strict']:.1f} / {100 * d['loose']:.1f}" if d else "n/a")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", f"n per cell: " + ", ".join(f"{m.split('-')[1]} {reg} {results['default'][(m, reg)]['n']}" for m, reg in keys if (m, reg) in results['default'])]
    text = "\n".join(lines)
    print(text)
    if args.markdown:
        with open(args.markdown, "w") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
