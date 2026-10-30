#!/usr/bin/env python3
"""L1 chance-match calibration: how often does the explanation search "explain" a number that cannot be a computation
over this item's context?

  P2_DATA_CACHE=... python3 -I calibrate_l1.py [--dataset finqa] [--n 300] [--markdown out.md]

For each sampled item, the probes are gold answers of other items (values unrelated to this context). A probe that the
search explains is a chance match. The rate is reported by magnitude and decimals so the paper can say which L1 labels
are credible (for example: decimals with 3+ digits rarely match by chance, small integers often do).
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detector import DetectorConfig, explain, matches_any  # noqa: E402
from loaders import load_dataset  # noqa: E402


def bucket(v: float) -> str:
    av = abs(v)
    if av == int(av):
        return "integer <= 12" if av <= 12 else "integer 13 to 999" if av < 1000 else "integer >= 1000"
    s = f"{av:.6f}".rstrip("0")
    dec = len(s.split(".")[1]) if "." in s else 0
    return f"decimal, {min(dec, 3)}{'+' if dec >= 3 else ''} decimals"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="finqa")
    ap.add_argument("--n", type=int, default=200, help="items sampled")
    ap.add_argument("--probes", type=int, default=5, help="probes per item")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args()
    items = [it for it in load_dataset(args.dataset) if it.gold_answer is not None]
    rng = random.Random(args.seed)
    sample = rng.sample(items, min(args.n, len(items)))
    golds = [it.gold_answer for it in items]
    cfg = DetectorConfig()
    tried: dict[str, int] = defaultdict(int)
    matched: dict[str, int] = defaultdict(int)
    for it in sample:
        probes = rng.sample(golds, args.probes)
        for v in probes:
            if v == 0 or matches_any([v], it.context_numbers, cfg):
                continue  # a copy, not a candidate for L1
            b = bucket(v)
            tried[b] += 1
            if explain([v], it.context_numbers, cfg):
                matched[b] += 1
    lines = [f"# L1 chance-match calibration on {args.dataset} ({len(sample)} items x {args.probes} probes; probes are gold answers of other items)", "",
             "| probe magnitude | probes | explained by chance | rate % |", "|---|---|---|---|"]
    tot_t = tot_m = 0
    for b in sorted(tried):
        tot_t += tried[b]
        tot_m += matched[b]
        lines.append(f"| {b} | {tried[b]} | {matched[b]} | {100 * matched[b] / tried[b]:.1f} |")
    lines.append(f"| all | {tot_t} | {tot_m} | {100 * tot_m / max(1, tot_t):.1f} |")
    lines += ["", "Reading: an L1 label on a number in a bucket with a high chance rate is weak evidence of in-text computation; the paper reports L1 with this table beside it and may restrict L1 to buckets under a chosen chance rate."]
    text = "\n".join(lines)
    print(text)
    if args.markdown:
        with open(args.markdown, "w") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
