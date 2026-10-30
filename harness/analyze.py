#!/usr/bin/env python3
"""Tables from results.csv (standard library only).

  python3 -I analyze.py results.csv [--markdown out.md] [--boot 1000]

Sections: leakage by regime x pressure (and by K, by dataset); sub-type shares; paired McNemar tests across pressures
within a regime (shared items); the probe (followed the tool, hidden computation, reported the discrepancy); the
FAILURE condition (reported the failure versus computed); the wrong-source rate under DISTRACT; leakage versus error
(Mantel-Haenszel relative risk stratified by regime x pressure x K with bypass episodes as their own stratum, bootstrap
over items, original and counterfactual arms separately, NOTOOL and PERTURB excluded); the gate cost table; the
noise floor from repeated seeds; cost per 100 episodes; detector validation against the mock's scripted truth.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00), "claude-opus-5-5": (4.00, 20.00)}  # USD per million tokens (list), cached 2026-10-06
BATCH_FACTOR = 0.5  # every run used the Message Batches API at half the list price (see results/cost-log.csv)
LEAK_REGIMES = ("OPTIONAL", "REQUIRED", "CONTRACT", "GATE")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (p, centre - half, centre + half)


def pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.1f}"


def load(path: str) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _i(r: dict, k: str) -> int:
    v = r.get(k, "0")
    return int(v) if v not in ("", None) else 0


def table_leakage(rows: list[dict], by: tuple[str, ...] = ("regime", "pressure")) -> list[str]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[tuple(r[b] for b in by)].append(r)
    lines = ["| " + " | ".join(by) + " | n | strict leak % (95% CI) | loose leak % | arg leak % | bypass % | accuracy % | mean tool calls | mean retries |", "|" + "---|" * (len(by) + 8)]
    for key in sorted(groups):
        g = groups[key]
        n = len(g)
        p, lo, hi = wilson(sum(_i(r, "leaked_strict") for r in g), n)
        loose = sum(_i(r, "leaked_loose") for r in g) / n
        arg = sum(_i(r, "argument_leakage") for r in g) / n
        byp = sum(_i(r, "tool_bypass") for r in g) / n
        acc = sum(_i(r, "correct") for r in g) / n
        calls = sum(_i(r, "n_tool_calls") for r in g) / n
        retries = sum(_i(r, "gate_rejections") + _i(r, "contract_violations") for r in g) / n
        lines.append(f"| {' | '.join(key)} | {n} | {pct(p)} ({pct(lo)} to {pct(hi)}) | {pct(loose)} | {pct(arg)} | {pct(byp)} | {pct(acc)} | {calls:.2f} | {retries:.2f} |")
    return lines


def subtype_shares(rows: list[dict]) -> list[str]:
    lines = ["| regime | pressure | n | leaked (strict) | L1 in-text % | L2 altered % | L3 unexplained % | L4 laundered % | wrong-source copies (final) |", "|---|---|---|---|---|---|---|---|---|"]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["regime"] in LEAK_REGIMES:
            groups[(r["regime"], r["pressure"])].append(r)
    for key in sorted(groups):
        g = groups[key]
        tot = sum(_i(r, f"L{k}_final") for r in g for k in (1, 2, 3, 4))
        shares = [sum(_i(r, f"L{k}_final") for r in g) / tot if tot else float("nan") for k in (1, 2, 3, 4)]
        ws = sum(_i(r, "wrong_source_final") for r in g)
        lines.append(f"| {key[0]} | {key[1]} | {len(g)} | {sum(_i(r, 'leaked_strict') for r in g)} | {pct(shares[0])} | {pct(shares[1])} | {pct(shares[2])} | {pct(shares[3])} | {ws} |")
    return lines


def mcnemar(rows: list[dict], metric: str = "leaked_strict", base: str = "NEUTRAL") -> list[str]:
    """Paired test on shared items: NEUTRAL versus each other pressure within a regime and model (exact binomial p)."""
    lines = [f"| regime | model | pressure vs {base} | items paired | {metric} base % | {metric} pressure % | discordant (b, c) | McNemar exact p |", "|---|---|---|---|---|---|---|---|"]
    by: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        if r["pressure"] == "PERTURB":
            continue
        by[(r["regime"], r["model"], r["pressure"], r["seed"])][r["item_id"]] = r
    keys = sorted({(k[0], k[1]) for k in by})
    for regime, model in keys:
        pressures = sorted({k[2] for k in by if k[0] == regime and k[1] == model and k[2] != base})
        for p in pressures:
            b = c = n = 0
            sb = sp = 0
            for seed in sorted({k[3] for k in by if k[0] == regime and k[1] == model}):
                base_rows = by.get((regime, model, base, seed), {})
                pr_rows = by.get((regime, model, p, seed), {})
                for item, rb in base_rows.items():
                    rp = pr_rows.get(item)
                    if rp is None:
                        continue
                    n += 1
                    xb, xp = _i(rb, metric), _i(rp, metric)
                    sb += xb
                    sp += xp
                    if xb == 1 and xp == 0:
                        b += 1
                    elif xb == 0 and xp == 1:
                        c += 1
            if n == 0:
                continue
            m = b + c
            pval = sum(math.comb(m, k) for k in range(0, min(b, c) + 1)) / (2 ** m) * 2 if m else 1.0
            pval = min(1.0, pval)
            lines.append(f"| {regime} | {model} | {p} | {n} | {pct(sb / n)} | {pct(sp / n)} | ({b}, {c}) | {pval:.3f} |")
    return lines


def probe_table(rows: list[dict]) -> list[str]:
    g = [r for r in rows if r["pressure"] == "PERTURB" and r["regime"] != "NOTOOL" and _i(r, "perturbed") == 1]
    if not g:
        return ["(no PERTURB episodes with a perturbed result)"]
    lines = ["| regime | model | n (perturbed) | followed the tool % | hidden computation % | reported the discrepancy % | correct % |", "|---|---|---|---|---|---|---|"]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in g:
        groups[(r["regime"], r["model"])].append(r)
    for key in sorted(groups):
        gg = groups[key]
        n = len(gg)
        lines.append(f"| {key[0]} | {key[1]} | {n} | {pct(sum(_i(r, 'followed_tool') for r in gg) / n)} | {pct(sum(_i(r, 'hidden_computation') for r in gg) / n)} | {pct(sum(_i(r, 'reported_discrepancy') for r in gg) / n)} | {pct(sum(_i(r, 'correct') for r in gg) / n)} |")
    return lines


def failure_table(rows: list[dict]) -> list[str]:
    g = [r for r in rows if r["pressure"] == "FAILURE" and r["regime"] != "NOTOOL"]
    if not g:
        return ["(no FAILURE episodes)"]
    lines = ["| regime | model | n | reported the failure % | retried the tool (2+ calls) % | strict leak % | correct % |", "|---|---|---|---|---|---|---|"]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in g:
        groups[(r["regime"], r["model"])].append(r)
    for key in sorted(groups):
        gg = groups[key]
        n = len(gg)
        lines.append(f"| {key[0]} | {key[1]} | {n} | {pct(sum(_i(r, 'reported_failure') for r in gg) / n)} | {pct(sum(1 for r in gg if _i(r, 'n_tool_calls') >= 2) / n)} | {pct(sum(_i(r, 'leaked_strict') for r in gg) / n)} | {pct(sum(_i(r, 'correct') for r in gg) / n)} |")
    return lines


def relative_risk(rows: list[dict], n_boot: int = 1000, seed: int = 0, arm: str = "original", bypass_stratum: bool = True) -> list[str]:
    """P(wrong | leaked) / P(wrong | not leaked), Mantel-Haenszel over regime x pressure x K strata (bypass episodes as their own
    stratum), tool regimes only, PERTURB excluded, with a bootstrap over items."""
    want_cf = 1 if arm == "counterfactual" else 0
    pool = [r for r in rows if r["regime"] in LEAK_REGIMES and r["pressure"] != "PERTURB" and _i(r, "counterfactual") == want_cf]
    if not pool:
        return [f"(no {arm} episodes in the tool regimes)"]

    def stratum(r: dict) -> tuple:
        if not bypass_stratum:  # bypass is close to a cause of leakage (a mediator); report the estimate without it as well
            return (r["regime"], r["pressure"], r["k_bucket"])
        return (r["regime"], r["pressure"], r["k_bucket"], "bypass" if _i(r, "tool_bypass") else "called")

    def mh_rr(rs: list[dict]) -> float:
        strata: dict[tuple, list[dict]] = defaultdict(list)
        for r in rs:
            strata[stratum(r)].append(r)
        num = den = 0.0
        for g in strata.values():
            n = len(g)
            a = sum(1 for r in g if _i(r, "leaked_strict") == 1 and _i(r, "correct") == 0)
            b = sum(1 for r in g if _i(r, "leaked_strict") == 1 and _i(r, "correct") == 1)
            c = sum(1 for r in g if _i(r, "leaked_strict") == 0 and _i(r, "correct") == 0)
            d = sum(1 for r in g if _i(r, "leaked_strict") == 0 and _i(r, "correct") == 1)
            n1, n0 = a + b, c + d
            if n1 == 0 or n0 == 0:
                continue
            num += a * n0 / n
            den += c * n1 / n
        return num / den if den > 0 else float("nan")

    rr = mh_rr(pool)
    rng = random.Random(seed)
    by_item: dict[str, list[dict]] = defaultdict(list)
    for r in pool:
        by_item[r["item_id"]].append(r)
    ids = list(by_item)
    boots = []
    for _ in range(n_boot):
        sample = [r for i in rng.choices(ids, k=len(ids)) for r in by_item[i]]
        v = mh_rr(sample)
        if v == v:
            boots.append(v)
    boots.sort()
    lo = boots[int(0.025 * len(boots))] if boots else float("nan")
    hi = boots[max(0, int(0.975 * len(boots)) - 1)] if boots else float("nan")
    le = [r for r in pool if _i(r, "leaked_strict") == 1]
    nl = [r for r in pool if _i(r, "leaked_strict") == 0]
    w1 = sum(1 for r in le if _i(r, "correct") == 0) / len(le) if le else float("nan")
    w0 = sum(1 for r in nl if _i(r, "correct") == 0) / len(nl) if nl else float("nan")
    lc = sum(1 for r in le if _i(r, "correct") == 1) / len(le) if le else float("nan")
    lines = [f"Arm: {arm}. Episodes {len(pool)} over {len(ids)} items; leaked {len(le)} (wrong {pct(w1)}%), not leaked {len(nl)} (wrong {pct(w0)}%); leaked-and-correct share {pct(lc)}%.",
             f"Mantel-Haenszel pooled relative risk of a wrong answer given strict leakage: {rr:.2f} (bootstrap 95% interval {lo:.2f} to {hi:.2f}, {n_boot} resamples; strata regime x pressure x K{' x bypass' if bypass_stratum else ' (no bypass stratum)'})."]
    if arm == "counterfactual":
        orig = [r for r in rows if r["regime"] in LEAK_REGIMES and r["pressure"] != "PERTURB" and _i(r, "counterfactual") == 0]
        if orig:
            c1 = sum(_i(r, "n_tool_calls") for r in pool) / len(pool)
            c0 = sum(_i(r, "n_tool_calls") for r in orig) / len(orig)
            lines.append(f"Tool calls per episode (all tool regimes and pressures pooled, so the original arm includes GATE; the paper quotes OPTIONAL by tier, see reports/paper-numbers.md), counterfactual {c1:.2f} versus original {c0:.2f} (interpret the size of the recall effect only when these match).")
    return lines


def gate_cost_table(rows: list[dict]) -> list[str]:
    lines = ["| regime | pressure | model | n | retries / 100 | delivered after retry exhaustion % | accuracy of delivered % | rejected a correct reply % | rejections on prose only (answer line traceable) % | strict leak % (not a finding for GATE) | tool calls | output tokens | USD / 100 episodes (batch prices) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["regime"] in ("REQUIRED", "CONTRACT", "GATE") and r["pressure"] != "PERTURB":
            groups[(r["regime"], r["pressure"], r["model"])].append(r)
    for key in sorted(groups):
        g = groups[key]
        n = len(g)
        retries = sum(_i(r, "gate_rejections") + _i(r, "contract_violations") for r in g) / n * 100
        exhausted = sum(1 for r in g if _i(r, "gate_rejections") >= 2 and _i(r, "leaked_strict") == 1 or _i(r, "contract_forced") == 1) / n
        acc = sum(_i(r, "correct") for r in g) / n
        has_scored = any("rejected_correct_n" in r for r in g)
        if has_scored:
            fr = sum(1 for r in g if _i(r, "rejected_correct_n") > 0) / n  # the rejected reply itself had the correct answer
            prose_only = sum(1 for r in g if _i(r, "rejected_strict_clean_n") > 0) / n
        else:
            fr = sum(1 for r in g if _i(r, "gate_rejections") > 0 and _i(r, "correct") == 1) / n  # proxy
            prose_only = float("nan")
        strict = sum(_i(r, "leaked_strict") for r in g) / n
        calls = sum(_i(r, "n_tool_calls") for r in g) / n
        otk = sum(_i(r, "output_tokens") for r in g) / n
        cost = sum(episode_cost(r) for r in g) / n * 100
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {n} | {retries:.1f} | {pct(exhausted)} | {pct(acc)} | {pct(fr)} | {pct(prose_only)} | {pct(strict)} | {calls:.2f} | {otk:.0f} | {cost:.2f} |")
    lines.append("")
    lines.append("'rejected a correct reply' scores the rejected reply itself (its answer line against the gold) when the run or the replay recorded rejected_correct_n; otherwise it is the proxy 'a rejection occurred and the delivered answer is correct'.")
    return lines


def episode_cost(r: dict) -> float:
    pi, po = PRICES.get(r["model"], (0.0, 0.0))
    return (_i(r, "input_tokens") / 1e6 * pi + _i(r, "output_tokens") / 1e6 * po) * BATCH_FACTOR


def noise_floor(rows: list[dict]) -> list[str]:
    by: dict[tuple, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[(r["regime"], r["pressure"], r["model"])][r["seed"]].append(_i(r, "leaked_strict"))
    lines = ["| cell | seeds | strict leak % per seed | range (points) |", "|---|---|---|---|"]
    found = False
    for key, seeds in sorted(by.items()):
        if len(seeds) < 2:
            continue
        found = True
        rates = [sum(v) / len(v) for v in seeds.values()]
        lines.append(f"| {' / '.join(key)} | {len(seeds)} | {', '.join(pct(x) for x in rates)} | {100 * (max(rates) - min(rates)):.1f} |")
    return lines if found else ["(no repeated seeds: run with --repeats 3 on one cell for the noise floor)"]


def detector_validation(rows: list[dict]) -> list[str]:
    mock = [r for r in rows if r["model"].startswith("mock")]
    if not mock:
        return ["(no mock rows: validation against scripted truth needs a dry run)"]
    tp = fp = fn = tn = 0
    for r in mock:
        truth = _i(r, "truth_emitted_leaks_n") > 0 or _i(r, "truth_altered") == 1 or _i(r, "truth_laundered") == 1
        pred = _i(r, "leaked_loose") == 1
        if truth and pred:
            tp += 1
        elif truth and not pred:
            fn += 1
        elif pred and not truth:
            fp += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return [f"Episode-level detector validation against the mock's scripted truth (loose metric): TP {tp}, FP {fp}, FN {fn}, TN {tn}; precision {pct(prec)}%, recall {pct(rec)}% over {len(mock)} mock episodes. This validates the implementation against its own definition; the probe and the counterfactual arm are the construct checks."]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--markdown", default=None)
    ap.add_argument("--boot", type=int, default=500)
    args = ap.parse_args()
    rows = load(args.csv)
    main_rows = [r for r in rows if _i(r, "counterfactual") == 0]
    out: list[str] = [f"# Analysis of {os.path.basename(args.csv)} ({len(rows)} episodes, {len(rows) - len(main_rows)} counterfactual)", ""]
    out += ["## Leakage by regime x pressure (original items)", ""] + table_leakage(main_rows) + [""]
    out += ["## Leakage by regime x operation bucket", ""] + table_leakage(main_rows, ("regime", "k_bucket")) + [""]
    out += ["## Leakage by dataset x regime", ""] + table_leakage(main_rows, ("dataset", "regime")) + [""]
    out += ["## Leakage by model x regime", ""] + table_leakage(main_rows, ("model", "regime")) + [""]
    out += ["## Sub-type shares among strict leaks, and wrong-source copies", ""] + subtype_shares(main_rows) + [""]
    out += ["## Paired tests across pressures (shared items, within regime and model)", ""] + mcnemar(main_rows) + [""]
    out += ["## Tool-perturbation probe", ""] + probe_table(main_rows) + [""]
    out += ["## After a calculator failure", ""] + failure_table(main_rows) + [""]
    out += ["## Does leakage predict errors?", ""] + relative_risk(rows, n_boot=args.boot, arm="original") + relative_risk(rows, n_boot=args.boot, arm="original", bypass_stratum=False)[1:2] + [""] + relative_risk(rows, n_boot=args.boot, arm="counterfactual") + relative_risk(rows, n_boot=args.boot, arm="counterfactual", bypass_stratum=False)[1:2] + [""]
    out += ["## The price of the gate (and the contract)", ""] + gate_cost_table(main_rows) + [""]
    out += ["## Noise floor (repeated seeds)", ""] + noise_floor(main_rows) + [""]
    out += ["## Detector validation", ""] + detector_validation(rows) + [""]
    text = "\n".join(out)
    print(text)
    if args.markdown:
        with open(args.markdown, "w") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
