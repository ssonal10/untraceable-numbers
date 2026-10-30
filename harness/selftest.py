#!/usr/bin/env python3
"""Zero-cost self-tests of the detector and the scoring rule over every FinQA test and TAT-QA dev item.

  P2_DATA_CACHE=/path/to/cache python3 -I selftest.py [--markdown selftest-report.md]

Tests (each reports a failure count and examples; the paper quotes the counts):
  T1  context numbers are never flagged: a reply that restates the item's own numbers has zero leaks
  T2  a gold-faithful reply is never flagged: the gold program is replayed as tool calls and the reply
      restates every intermediate and the gold answer
  T3  gold program arguments are never argument leaks (same replay, argument_leakage must be false)
  T4  the scoring rule reproduces the dataset's own answer strings (FinQA `answer` versus `exe_ans`;
      TAT-QA `answer` versus the evaluated derivation with its scale)
  T5  collision share: items whose gold answer or a gold intermediate already appears in the context or
      the constant set (unflaggable by construction; the paper reports this share and a sensitivity row)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calculator import Calculator, format_number  # noqa: E402
from detector import DetectorConfig, Episode, ToolCall, answer_is_correct, decimals_in, detect, extract_mentions, matches_any  # noqa: E402
from loaders import load_dataset  # noqa: E402


def replay_gold(item, calc: Calculator) -> tuple[list[ToolCall], list[str], bool]:
    """Run the gold program through the calculator; return the tool calls, the step lines, and whether it resolved."""
    calls: list[ToolCall] = []
    lines: list[str] = []
    ids: dict[int, str] = {}
    for i, step in enumerate(item.gold_steps):
        if not step.expr or step.value is None:
            return calls, lines, False
        expr = step.expr
        for ref in re.findall(r"#(\d+)", expr):
            expr = expr.replace(f"#{ref}", ids.get(int(ref), "0"))
        res = calc.call(expr)
        if res.error or res.value is None:
            return calls, lines, False
        ids[i] = res.id
        args = [v for v in extract_mentions(expr, "text", DetectorConfig(exclude_list_markers=False))]
        arg_vals = [a.value for a in args] + [calc.results[r] for r in re.findall(r"\br\d+\b", expr) if r in calc.results]
        calls.append(ToolCall(name="calculator", arg_numbers=arg_vals, output_value=res.value, expression=expr))
        lines.append(f"Step {i + 1}: {res.id} = {format_number(res.value)}.")
    return calls, lines, True


def episode(item, calls, texts, final, available=True) -> Episode:
    return Episode(context_numbers=item.context_numbers, tool_calls=calls, output_texts=texts, final_answer_text=final,
                   tool_available=available, gold_needs_arithmetic=item.needs_arithmetic, gold_intermediates=item.gold_intermediates,
                   gold_answer=item.gold_answer, context_years=item.context_years)


def parse_answer_string(s) -> float | None:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    t = str(s).strip().replace(",", "").replace("$", "")
    neg = t.startswith("(") and t.endswith(")")
    t = t.strip("()").replace("%", "").strip()
    try:
        v = float(t)
    except ValueError:
        return None
    return -v if neg else v


def run(dataset: str, max_items: int | None = None) -> dict:
    items = load_dataset(dataset)
    if max_items:
        items = items[:max_items]
    cfg = DetectorConfig()
    out = {"dataset": dataset, "n": len(items), "T1_fail": 0, "T2_fail": 0, "T3_fail": 0, "T4_fail": 0, "T4_n": 0,
           "unresolved": 0, "collision_items": 0, "examples": {"T1": [], "T2": [], "T3": [], "T4": []}}
    for it in items:
        # T1: restate the item's own numbers (first 40 numerals, as the detector reads them; a range such as "1-4 family"
        # is read as 1 and 4, so the restated form is the value, not the raw substring)
        raws = [format_number(m.value) for m in extract_mentions(it.context_text, "text", DetectorConfig(exclude_list_markers=False))][:40]
        reply = "The figures are " + ", ".join(raws) + ". Answer: " + (raws[-1] if raws else "0")
        d = detect(episode(it, [], [reply], reply), cfg)
        if d.leaked_loose:
            out["T1_fail"] += 1
            if len(out["examples"]["T1"]) < 3:
                out["examples"]["T1"].append((it.id, [m.raw + ":" + m.label for m in d.mentions if m.label in ("L1", "L2", "L3", "L4")][:3]))
        # T2 and T3: replay the gold program
        calc = Calculator()
        calls, lines, ok = replay_gold(it, calc)
        if not ok:
            out["unresolved"] += 1
        else:
            final_val = calls[-1].output_value
            final = " ".join(lines) + f" Answer: {format_number(final_val)}"
            d2 = detect(episode(it, calls, [final], final), cfg)
            if d2.leaked_strict or d2.leaked_loose:
                out["T2_fail"] += 1
                if len(out["examples"]["T2"]) < 3:
                    out["examples"]["T2"].append((it.id, [m.raw + ":" + m.label + ":" + m.explanation[:30] for m in d2.mentions if m.label in ("L1", "L2", "L3", "L4")][:3]))
            if d2.argument_leakage:
                out["T3_fail"] += 1
                if len(out["examples"]["T3"]) < 3:
                    out["examples"]["T3"].append((it.id, [c.expression for c in calls][:2]))
        # T4: scoring rule against the dataset's own answer string
        ans_raw = str(it.raw.get("answer")) if dataset == "finqa" else str(it.gold_answer)
        ans = parse_answer_string(it.raw.get("answer")) if dataset == "finqa" else it.gold_answer
        gold = it.gold_answer if dataset == "finqa" else (it.gold_steps[0].value * (100.0 if it.answer_scale == "percent" else 1.0) if it.gold_steps and it.gold_steps[0].value is not None else None)
        if ans is not None and gold is not None:
            out["T4_n"] += 1
            if not answer_is_correct(ans, gold, pred_decimals=decimals_in(ans_raw)):
                out["T4_fail"] += 1
                if len(out["examples"]["T4"]) < 3:
                    out["examples"]["T4"].append((it.id, ans, gold))
        # T5: collisions
        gold_refs = list(it.gold_intermediates) + ([it.gold_answer] if it.gold_answer is not None else [])
        consts = list(cfg.constant_args)
        if gold_refs and (matches_any(gold_refs, it.context_numbers, cfg) or any(abs(g - c) < 1e-9 for g in gold_refs for c in consts)):
            out["collision_items"] += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="finqa,tatqa")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args()
    lines = ["# Detector self-tests over the public items (zero API cost)", ""]
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        r = run(ds, args.max_items)
        n = r["n"]
        lines += [f"## {ds}: {n} items ({r['unresolved']} whose gold program does not evaluate from the file; excluded from T2/T3)", "",
                  f"- T1 context numbers never flagged: {r['T1_fail']} failures / {n}",
                  f"- T2 gold-faithful reply never flagged: {r['T2_fail']} failures / {n - r['unresolved']}",
                  f"- T3 gold program arguments never argument leaks: {r['T3_fail']} failures / {n - r['unresolved']}",
                  f"- T4 scoring rule reproduces the dataset's answer string: {r['T4_fail']} failures / {r['T4_n']}",
                  f"- T5 collision share (gold answer or intermediate already in the context or the constant set): {r['collision_items']} / {n} = {100 * r['collision_items'] / max(1, n):.1f} percent", ""]
        for t, ex in r["examples"].items():
            for e in ex:
                lines.append(f"  - {t} example: {e}")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if args.markdown:
        with open(args.markdown, "w") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
