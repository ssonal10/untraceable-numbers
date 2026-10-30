#!/usr/bin/env python3
"""Labeling export and agreement.

Export (stratified sample for human validation):
  python3 -I export_labels.py export results.csv --traces traces.jsonl --flagged 120 --unflagged 40 --out labels.csv

  Writes one row per flagged mention (stratified over regime x pressure x leak type) plus one row per unflagged episode
  (final answer with no leak), with the detector's label and explanation pre-filled, the reply text (from traces when given,
  else the truncated final_text), and empty columns `label_A`, `label_B`, `type_A`, `type_B`, `note` for two annotators.
  Label values: LEAK or OK; type values: L1, L2, L3, L4, COPY, TOOL, OTHER.

Agreement:
  python3 -I export_labels.py kappa labels.csv [--col-a label_A --col-b label_B]

  Prints Cohen's kappa, raw agreement, and the detector's precision and recall against annotator A (and B).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_mentions(s: str) -> list[dict]:
    """The CSV `mentions` column is capped at 2,000 characters; parse what is complete (traces hold the full list)."""
    try:
        return json.loads(s or "[]")
    except json.JSONDecodeError:
        cut = s.rfind("}")
        if cut < 0:
            return []
        try:
            return json.loads(s[: cut + 1] + "]")
        except json.JSONDecodeError:
            return []


def load_traces(path: str | None) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    if not path or not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            try:
                t = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[(t["item_id"], t["cond"], str(t.get("seed", 0)))] = t
    return out


def cmd_export(args) -> int:
    rows = list(csv.DictReader(open(args.csv)))
    traces = load_traces(args.traces)
    rng = random.Random(args.seed)
    flagged: dict[tuple, list[dict]] = defaultdict(list)
    unflagged: list[dict] = []
    for r in rows:
        if r["regime"] == "NOTOOL":
            continue
        cond = f"{r['regime']}|{r['pressure']}|{r['effort']}|{r['fmt']}" + ("" if r.get("temperature", "") == "" else f"|t{r['temperature']}")
        key = (r["item_id"], cond, r["seed"])
        reply = traces.get(key, {}).get("final_text") or r.get("final_text", "")
        texts = " ".join(traces.get(key, {}).get("output_texts", [])) if key in traces else ""
        mentions = traces.get(key, {}).get("mentions") or parse_mentions(r.get("mentions") or "[]")
        mentions = [{"raw": m.get("raw"), "where": m.get("where"), "label": m.get("label"), "why": m.get("why", m.get("explanation", ""))} for m in mentions]
        leaks = [m for m in mentions if m["label"] in ("L1", "L2", "L3", "L4")]
        if leaks:
            for m in leaks:
                flagged[(r["regime"], r["pressure"], m["label"])].append({**r, "_m": m, "_reply": reply, "_texts": texts})
        elif r.get("leaked_loose", "0") == "0":
            unflagged.append({**r, "_reply": reply, "_texts": texts})
    # stratified allocation: equal shares per (regime, pressure, type) stratum, then fill from the largest strata
    strata = sorted(flagged)
    picked: list[dict] = []
    if strata:
        per = max(1, args.flagged // len(strata))
        for s in strata:
            pool = flagged[s]
            rng.shuffle(pool)
            picked.extend(pool[:per])
        rest = [x for s in strata for x in flagged[s][per:]]
        rng.shuffle(rest)
        picked.extend(rest[: max(0, args.flagged - len(picked))])
    rng.shuffle(unflagged)
    picked_un = unflagged[: args.unflagged]
    fields = ["row_type", "item_id", "dataset", "regime", "pressure", "model", "seed", "mention", "where", "detector_label", "detector_explanation",
              "final_text", "output_texts", "pred", "gold", "correct", "label_A", "type_A", "label_B", "type_B", "note"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for x in picked:
            m = x["_m"]
            w.writerow({"row_type": "flagged", "item_id": x["item_id"], "dataset": x["dataset"], "regime": x["regime"], "pressure": x["pressure"], "model": x["model"], "seed": x["seed"],
                        "mention": m["raw"], "where": m["where"], "detector_label": m["label"], "detector_explanation": m.get("why", ""),
                        "final_text": x["_reply"], "output_texts": x["_texts"][:1500], "pred": x.get("pred", ""), "gold": x.get("gold", ""), "correct": x.get("correct", "")})
        for x in picked_un:
            w.writerow({"row_type": "unflagged", "item_id": x["item_id"], "dataset": x["dataset"], "regime": x["regime"], "pressure": x["pressure"], "model": x["model"], "seed": x["seed"],
                        "mention": "", "where": "final", "detector_label": "OK", "detector_explanation": "",
                        "final_text": x["_reply"], "output_texts": x["_texts"][:1500], "pred": x.get("pred", ""), "gold": x.get("gold", ""), "correct": x.get("correct", "")})
    print(f"wrote {len(picked)} flagged mentions over {len(strata)} strata and {len(picked_un)} unflagged episodes to {args.out}")
    return 0


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] / n * cb[k] / n for k in set(ca) | set(cb))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def cmd_kappa(args) -> int:
    rows = [r for r in csv.DictReader(open(args.csv)) if r.get(args.col_a) and r.get(args.col_b)]
    if not rows:
        print("no rows with both annotator labels")
        return 1
    a = [r[args.col_a].strip().upper() for r in rows]
    b = [r[args.col_b].strip().upper() for r in rows]
    print(f"rows with both labels: {len(rows)}; raw agreement {100 * sum(1 for x, y in zip(a, b) if x == y) / len(rows):.1f}%; Cohen's kappa {cohen_kappa(a, b):.3f}")
    for col, lab in ((args.col_a, a), (args.col_b, b)):
        det = ["LEAK" if r["detector_label"] in ("L1", "L2", "L3", "L4") else "OK" for r in rows]
        tp = sum(1 for d, h in zip(det, lab) if d == "LEAK" and h == "LEAK")
        fp = sum(1 for d, h in zip(det, lab) if d == "LEAK" and h != "LEAK")
        fn = sum(1 for d, h in zip(det, lab) if d != "LEAK" and h == "LEAK")
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        print(f"detector versus {col}: precision {100 * prec:.1f}%, recall {100 * rec:.1f}% (on this stratified sample, not population rates)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("csv")
    e.add_argument("--traces", default=None)
    e.add_argument("--flagged", type=int, default=120)
    e.add_argument("--unflagged", type=int, default=40)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--out", default="labels.csv")
    k = sub.add_parser("kappa")
    k.add_argument("csv")
    k.add_argument("--col-a", default="label_A")
    k.add_argument("--col-b", default="label_B")
    args = ap.parse_args()
    return cmd_export(args) if args.cmd == "export" else cmd_kappa(args)


if __name__ == "__main__":
    sys.exit(main())
