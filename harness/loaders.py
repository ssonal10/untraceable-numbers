"""Dataset loaders for FinQA, TAT-QA and ConvFinQA in their original repository formats.

Formats were confirmed on the public files on 2026-10-20:
  FinQA      github.com/czyssrs/FinQA           dataset/test.json  (1,147 items; qa.program, qa.exe_ans)
  ConvFinQA  github.com/czyssrs/ConvFinQA       data.zip -> data/dev.json (421 conversations;
             annotation.dialogue_break / turn_program / exe_ans_list)
  TAT-QA     github.com/NExTplusplus/TAT-QA     dataset_raw/tatqa_dataset_dev.json (278 contexts,
             1,668 questions, 718 arithmetic with `derivation` and `scale`)

Files are fetched into a cache directory (never redistributed). If the Hugging Face
`datasets` library is installed and P2_HF_<DATASET> names a mirror, that mirror is used
instead; mirror ids are not verified, so the repository files are the default path.

Every item becomes an Item with: id, dataset, question, context_text, context_numbers,
context_years, history (ConvFinQA turns), gold_answer, gold_steps (resolved operations),
k (operation count), needs_arithmetic.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import re
import sys
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from calculator import Calculator, clean_expression
from detector import extract_numbers

URLS = {
    # pinned to the commits whose files the study used (blob hashes checked on 2026-10-29); SHA-256 below
    "finqa": "https://raw.githubusercontent.com/czyssrs/FinQA/0f16e2867befa6840783e58be38c9efb9229d742/dataset/{split}.json",
    "convfinqa": "https://github.com/czyssrs/ConvFinQA/raw/main/data.zip",
    "tatqa": "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/870accc41953dcde885aabeb963d94aabdc0fbc3/dataset_raw/tatqa_dataset_{split}.json",
}
SHA256 = {  # files used in the study; fetch() warns when a cached or downloaded file differs
    ("finqa", "test"): "831dbfb2e785dbc227f895ce3f24046433467aec67b09db2bd6ac7692a8a30dc",
    ("tatqa", "dev"): "8da095a819af6db3c14877c6df2d4d29960e41d1a63dd1fa853507bd2a616af5",
}
DEFAULT_SPLITS = {"finqa": "test", "convfinqa": "dev", "tatqa": "dev"}


@dataclass
class GoldStep:
    op: str
    a: float | None
    b: float | None
    value: float | None
    expr: str  # calculator expression with literals; '#k' references rendered later by the mock
    refs: tuple[int, ...] = ()  # indices of earlier steps this step references


@dataclass
class Item:
    id: str
    dataset: str
    question: str
    context_text: str
    context_numbers: list[float]
    context_years: set[int]
    gold_answer: float | None
    gold_steps: list[GoldStep]
    k: int
    needs_arithmetic: bool
    history: list[tuple[str, str]] = field(default_factory=list)  # (question, gold answer text) for ConvFinQA
    answer_scale: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def k_bucket(self) -> str:
        return "0" if self.k == 0 else "1" if self.k == 1 else "2" if self.k == 2 else "3+"

    @property
    def gold_intermediates(self) -> list[float]:
        vals = [s.value for s in self.gold_steps if s.value is not None]
        if self.answer_scale == "percent" and vals:
            vals = vals + [v * 100.0 for v in vals]
        return vals


# ----------------------------------------------------------------------------- fetching

def cache_dir() -> Path:
    d = Path(os.environ.get("P2_DATA_CACHE", Path.home() / ".cache" / "p2-leakage"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch(dataset: str, split: str | None = None, cache: Path | None = None) -> Path:
    """Download the public file into the cache (once) and return its path."""
    cache = cache or cache_dir()
    split = split or DEFAULT_SPLITS[dataset]
    sub = cache / dataset
    sub.mkdir(parents=True, exist_ok=True)
    if dataset == "convfinqa":
        target = sub / "data" / f"{split}.json"
        if not target.exists():
            zpath = sub / "data.zip"
            if not zpath.exists():
                urllib.request.urlretrieve(URLS[dataset], zpath)
            with zipfile.ZipFile(zpath) as zf:
                zf.extract(f"data/{split}.json", sub)
        return target
    name = f"{split}.json" if dataset == "finqa" else f"tatqa_dataset_{split}.json"
    target = sub / name
    if not target.exists():
        urllib.request.urlretrieve(URLS[dataset].format(split=split), target)
    want = SHA256.get((dataset, split))
    if want:
        got = hashlib.sha256(target.read_bytes()).hexdigest()
        if got != want:
            print(f"warning: {target} has SHA-256 {got}, not the file used in the study ({want})", file=sys.stderr)
    return target


# ----------------------------------------------------------------------------- FinQA program DSL

_STEP_RE = re.compile(r"(\w+)\(([^,]+),\s*([^)]*)\)")
_CONST_RE = re.compile(r"^const_(m?)(\d+)$")


def _operand(tok: str, prior: list[GoldStep], table: list[list[str]] | None) -> tuple[float | None, str, tuple[int, ...]]:
    tok = tok.strip()
    if tok.startswith("#"):
        idx = int(tok[1:])
        val = prior[idx].value if idx < len(prior) else None
        return val, f"#{idx}", (idx,)
    m = _CONST_RE.match(tok)
    if m:
        v = float(m.group(2)) * (-1 if m.group(1) else 1)
        return v, str(v), ()
    if tok in ("none", ""):
        return None, "", ()
    try:
        v = float(tok.replace(",", "").replace("%", "").replace("$", ""))
        return v, tok.replace(",", ""), ()
    except ValueError:
        return None, tok, ()


def _table_row_values(table: list[list[str]], header: str) -> list[float]:
    for row in table or []:
        if row and row[0].strip().lower() == header.strip().lower():
            vals = []
            for cell in row[1:]:
                nums = extract_numbers(cell)
                if nums:
                    vals.append(nums[0])
            return vals
    return []


def parse_finqa_program(program: str, table: list[list[str]] | None = None) -> list[GoldStep]:
    steps: list[GoldStep] = []
    for op, a_tok, b_tok in _STEP_RE.findall(program or ""):
        a, a_expr, a_refs = _operand(a_tok, steps, table)
        b, b_expr, b_refs = _operand(b_tok, steps, table)
        value: float | None = None
        expr = ""
        try:
            if op == "add" and a is not None and b is not None:
                value, expr = a + b, f"{a_expr} + {b_expr}"
            elif op == "subtract" and a is not None and b is not None:
                value, expr = a - b, f"{a_expr} - {b_expr}"
            elif op == "multiply" and a is not None and b is not None:
                value, expr = a * b, f"{a_expr} * {b_expr}"
            elif op == "divide" and a is not None and b is not None and b != 0:
                value, expr = a / b, f"{a_expr} / {b_expr}"
            elif op == "exp" and a is not None and b is not None:
                value, expr = a ** b, f"{a_expr} ** {b_expr}"
            elif op == "greater" and a is not None and b is not None:
                value, expr = (1.0 if a > b else 0.0), f"max({a_expr}, {b_expr})"
            elif op in ("table_sum", "table_average", "table_max", "table_min"):
                vals = _table_row_values(table or [], a_tok)
                if vals:
                    fn = {"table_sum": sum, "table_average": lambda v: sum(v) / len(v), "table_max": max, "table_min": min}[op]
                    value = float(fn(vals))
                    name = {"table_sum": "sum", "table_average": "avg", "table_max": "max", "table_min": "min"}[op]
                    expr = f"{name}({', '.join(str(v) for v in vals)})"
        except (OverflowError, ZeroDivisionError, ValueError):
            value = None
        steps.append(GoldStep(op=op, a=a, b=b, value=value, expr=expr, refs=tuple(a_refs + b_refs)))
    return steps


# ----------------------------------------------------------------------------- rendering

def render_table(table: list[list[str]]) -> str:
    lines = []
    for i, row in enumerate(table or []):
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
        if i == 0:
            lines.append("|" + "---|" * len(row))
    return "\n".join(lines)


def _context_from_parts(pre: list[str], table: list[list[str]], post: list[str]) -> str:
    parts = []
    if pre:
        parts.append("Text before the table:\n" + " ".join(pre))
    if table:
        parts.append("Table:\n" + render_table(table))
    if post:
        parts.append("Text after the table:\n" + " ".join(post))
    return "\n\n".join(parts)


def _years_in(text: str) -> set[int]:
    return {int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", text)}


def _to_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(",", "").replace("$", "").replace("%", "")
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return float(s)
    except ValueError:
        return None


# ----------------------------------------------------------------------------- loaders

def load_finqa(path: Path, limit: int | None = None) -> list[Item]:
    data = json.loads(Path(path).read_text())
    items: list[Item] = []
    for ex in data[: limit or None]:
        qa = ex["qa"]
        steps = parse_finqa_program(qa.get("program", ""), ex.get("table"))
        ctx = _context_from_parts(ex.get("pre_text", []), ex.get("table", []), ex.get("post_text", []))
        gold = _to_float(qa.get("exe_ans"))
        items.append(
            Item(
                id=f"finqa:{ex['id']}",
                dataset="finqa",
                question=qa["question"],
                context_text=ctx,
                context_numbers=extract_numbers(ctx + "\n" + qa["question"]),
                context_years=_years_in(ctx + " " + qa["question"]),
                gold_answer=gold,
                gold_steps=steps,
                k=len(steps),
                needs_arithmetic=len(steps) > 0,
                raw={"program": qa.get("program", ""), "answer": qa.get("answer"), "table": ex.get("table", [])},
            )
        )
    return items


def load_convfinqa(path: Path, limit: int | None = None) -> list[Item]:
    data = json.loads(Path(path).read_text())
    items: list[Item] = []
    for ex in data[: limit or None]:
        ann = ex["annotation"]
        ctx = _context_from_parts(ex.get("pre_text", []), ex.get("table", []), ex.get("post_text", []))
        questions = ann["dialogue_break"]
        programs = ann["turn_program"]
        answers = ann["exe_ans_list"]
        history: list[tuple[str, str]] = []
        for t, (q, prog, ans) in enumerate(zip(questions, programs, answers)):
            steps = parse_finqa_program(prog, ex.get("table")) if "(" in prog else []
            items.append(
                Item(
                    id=f"convfinqa:{ex['id']}:t{t}",
                    dataset="convfinqa",
                    question=q,
                    context_text=ctx,
                    context_numbers=extract_numbers(ctx + "\n" + q),
                    context_years=_years_in(ctx + " " + q),
                    gold_answer=_to_float(ans),
                    gold_steps=steps,
                    k=len(steps),
                    needs_arithmetic=len(steps) > 0,
                    history=list(history),
                    raw={"program": prog, "turn": t, "n_turns": len(questions)},
                )
            )
            history.append((q, str(ans)))
    return items


_DERIV_OPS = re.compile(r"[+\-*/]")


def _derivation_steps(derivation: str, calc: Calculator) -> tuple[list[GoldStep], int]:
    """Count binary operators and evaluate the derivation as one expression."""
    expr = derivation.replace("%", "")
    # operators that are not unary minus: count +, *, / and binary minus (preceded by a digit or ')')
    k = len(re.findall(r"[+*/]", expr)) + len(re.findall(r"(?<=[\d\)])\s*-\s*(?=[\d\(])", expr))
    value: float | None = None
    try:
        value = calc.evaluate(expr)
    except Exception:  # noqa: BLE001
        value = None
    steps = [GoldStep(op="derivation", a=None, b=None, value=value, expr=clean_expression(expr) if value is not None else "")]
    return steps, k


def load_tatqa(path: Path, limit: int | None = None) -> list[Item]:
    data = json.loads(Path(path).read_text())
    calc = Calculator()
    items: list[Item] = []
    for ctx_ex in data:
        table = ctx_ex["table"]["table"]
        paras = [p["text"] for p in sorted(ctx_ex["paragraphs"], key=lambda p: p.get("order", 0))]
        ctx = _context_from_parts([], table, paras)
        for q in ctx_ex["questions"]:
            if q.get("answer_type") != "arithmetic":
                continue
            steps, k = _derivation_steps(q.get("derivation", ""), calc)
            gold = _to_float(q["answer"] if not isinstance(q["answer"], list) else q["answer"][0])
            items.append(
                Item(
                    id=f"tatqa:{q['uid']}",
                    dataset="tatqa",
                    question=q["question"] + (f" (answer in {q['scale']})" if q.get("scale") else ""),
                    context_text=ctx,
                    context_numbers=extract_numbers(ctx + "\n" + q["question"]),
                    context_years=_years_in(ctx + " " + q["question"]),
                    gold_answer=gold,
                    gold_steps=steps,
                    k=max(k, 1),
                    needs_arithmetic=True,
                    answer_scale=q.get("scale", ""),
                    raw={"derivation": q.get("derivation", ""), "scale": q.get("scale", "")},
                )
            )
            if limit and len(items) >= limit:
                return items
    return items


LOADERS = {"finqa": load_finqa, "convfinqa": load_convfinqa, "tatqa": load_tatqa}


def load_dataset(dataset: str, path: Path | None = None, limit: int | None = None, cache: Path | None = None) -> list[Item]:
    if path is None:
        path = fetch(dataset, cache=cache)
    return LOADERS[dataset](Path(path), limit=limit)


def load_mini_sample(path: Path) -> list[Item]:
    """The bundled miniature sample (FinQA format, handmade numbers) for tests and offline dry runs."""
    return load_finqa(path)


# ----------------------------------------------------------------------------- counterfactual values

_CF_FACTORS = (0.83, 0.91, 1.07, 1.13, 1.21)


def _find_literal(text: str, v: float) -> tuple[re.Match | None, int]:
    """Locate the number v in the context text as written (with or without thousands separators). Returns (match, decimals)."""
    for decimals in (0, 1, 2, 3, 4):
        if abs(v - round(v, decimals)) > 1e-9:
            continue
        s = f"{abs(v):,.{decimals}f}"
        for form in (s, s.replace(",", "")):
            m = re.search(r"(?<![\d.,])" + re.escape(form) + r"(?![\d])", text)
            if m:
                return m, decimals
        break
    return None, 0


def counterfactual(item: Item, seed: int) -> Item | None:
    """Perturb the numbers the gold program reads, in the context and in the program, and recompute the gold answer.

    A correct answer on a counterfactual item cannot be recalled from pretraining data; a leaked number that is correct
    there was computed by the model itself. Returns None when a literal cannot be located in the context text.
    """
    rng = random.Random(f"cf:{seed}:{item.id}")
    text = item.context_text
    program = item.raw.get("program") or item.raw.get("derivation") or ""
    if not program:
        return None
    table = [list(r) for r in item.raw.get("table", [])]
    constants = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 100.0, 1000.0, 1e6, 1e9)  # same list as the detector
    literals: list[float] = []
    # table operations (table_sum, table_average, ...) read a whole row: perturb every numeric cell of that row
    table_rows_used = [s for s in item.gold_steps if s.op.startswith("table_")]
    for step in table_rows_used:
        header = re.match(r"\w+\(([^,]+),", program[program.find(step.op):]).group(1).strip() if step.op in program else ""
        row = next((r for r in table if r and r[0].strip().lower() == header.lower()), None)
        if row is None:
            return None
        for ci in range(1, len(row)):
            nums = extract_numbers(row[ci])
            if nums and nums[0] not in literals and nums[0] not in constants:
                literals.append(nums[0])
    for step in item.gold_steps:
        for v in (step.a, step.b):
            if v is not None and v not in literals and not any(abs(v - s.value) < 1e-9 for s in item.gold_steps if s.value is not None):
                if v in constants:  # constants, not context values
                    continue
                literals.append(v)
    if item.dataset == "tatqa":
        literals = [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.?\d*", program.replace("%", ""))]
        literals = [v for v in literals if v not in (100.0, 1000.0, 2.0, 3.0)]
    if not literals:
        return None
    mapping: dict[float, float] = {}
    new_text = text
    for v in literals:
        m, decimals = _find_literal(new_text, v)
        if m is None:
            return None
        nv = round(v * rng.choice(_CF_FACTORS), decimals)
        if nv == v:
            nv = round(v * 1.3, decimals)
        mapping[v] = nv
        new_form = f"{abs(nv):,.{decimals}f}" if "," in m.group(0) else f"{abs(nv):.{decimals}f}"
        new_text = new_text[: m.start()] + new_form + new_text[m.end():]
        # replace every other occurrence of the same literal form so the table and the text stay consistent
        new_text = re.sub(r"(?<![\d.,])" + re.escape(m.group(0)) + r"(?![\d])", new_form, new_text)
        for row in table:  # keep the raw table in step so table operations re-evaluate on the perturbed values
            for ci in range(len(row)):
                row[ci] = re.sub(r"(?<![\d.,])" + re.escape(m.group(0)) + r"(?![\d])", new_form, row[ci])
    new_program = program
    for v, nv in sorted(mapping.items(), key=lambda kv: -len(str(kv[0]))):
        for form in (f"{v:,.4f}".rstrip("0").rstrip("."), f"{v:.4f}".rstrip("0").rstrip("."), str(v), str(int(v)) if v == int(v) else None):
            if form:
                new_program = re.sub(r"(?<![\d.])" + re.escape(form) + r"(?![\d])", f"{nv:.4f}".rstrip("0").rstrip("."), new_program)
    if item.dataset == "tatqa":
        calc = Calculator()
        steps, k = _derivation_steps(new_program, calc)
        gold = steps[0].value
        if gold is None:
            return None
        if item.answer_scale == "percent":
            gold = gold * 100.0
    else:
        steps = parse_finqa_program(new_program, table or None)
        if not steps or steps[-1].value is None or not all(s.value is not None and s.expr for s in steps):
            return None
        gold = steps[-1].value
    new_item = Item(
        id=item.id + ":cf", dataset=item.dataset, question=item.question, context_text=new_text,
        context_numbers=extract_numbers(new_text + "\n" + item.question), context_years=item.context_years,
        gold_answer=gold, gold_steps=steps, k=item.k, needs_arithmetic=item.needs_arithmetic, history=[],
        answer_scale=item.answer_scale, raw={**item.raw, "table": table, "counterfactual": True, "program_cf": new_program, "mapping": mapping},
    )
    return new_item


# ----------------------------------------------------------------------------- sampling

def is_fully_resolved(it: Item) -> bool:
    """Every gold step evaluated to a number and has a calculator expression (needed by the scripted mock)."""
    return all(s.value is not None and s.expr for s in it.gold_steps)


def stratified_sample(items: list[Item], n_per_bucket: int, seed: int, buckets: tuple[str, ...] = ("1", "2", "3+"), resolved_only: bool = True) -> list[Item]:
    rng = random.Random(seed)
    by_bucket: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        if it.gold_answer is None:
            continue
        if resolved_only and not is_fully_resolved(it):
            continue
        by_bucket[it.k_bucket].append(it)
    out: list[Item] = []
    for b in buckets:
        pool = sorted(by_bucket.get(b, []), key=lambda it: it.id)
        rng.shuffle(pool)
        out.extend(pool[:n_per_bucket])
    return out


def bucket_counts(items: list[Item]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for it in items:
        counts[it.k_bucket] += 1
    return dict(sorted(counts.items()))
