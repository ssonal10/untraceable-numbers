"""Arithmetic-leakage detector: regex extraction plus trace comparison.

Input: an Episode record (context numbers, tool trace, model output texts,
final answer text, optional gold intermediates). Output: one label per number
mention in the output, and episode-level rates.

Labels for a number mention x:
  COPY_CTX     x matches a number in the context (table, text, question)
  TOOL_OUT     x matches a value returned by a tool call in this episode
  TOOL_IN_ECHO x matches a tool argument that was itself traceable (a restated input)
  L1           in-text computation: x matches no tool output, and a pair of
               reference numbers plus one operation reproduces x
  L2           altered tool output: x is near a tool output (within 10 percent,
               or one digit changed, or a moved decimal point) but outside tolerance
  L3           unexplained: none of the above
  EXCLUDED     year, list marker, step number, or an explicit exclusion

Episode flags: leaked_strict (any L1/L2/L3 in the final answer), leaked_loose
(any L1/L2/L3 anywhere in the output), argument_leakage (a tool argument that
matches neither the context nor an earlier tool output), tool_bypass (gold needs
arithmetic, tool available, no tool call).

Standard library only. All thresholds are parameters of DetectorConfig so the
paper can report a sensitivity table.
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

# A number mention: optional currency, optional parentheses, digits with optional
# thousands separators and decimals, optional percent, optional scale word.
_NUM_RE = re.compile(
    r"(?P<paren>\()?"
    r"(?:(?P<cur>[\$€£])\s?)?"
    r"(?P<sign>[-−])?"
    r"(?P<body>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
    r"(?P<pct>\s?%|\s?percent)?"
    r"(?(paren)\))"
    r"(?:\s?(?P<scale>thousand|million|billion|trillion|bn|mn|k|m|b)\b)?",
    re.IGNORECASE,
)
_SCALE = {"thousand": 1e3, "k": 1e3, "million": 1e6, "mn": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9, "b": 1e9, "trillion": 1e12}
_LIST_MARKER_RE = re.compile(r"(?:^|\n)\s*(?:step\s*)?(\d{1,2})[\.\):]\s", re.IGNORECASE)
_RESULT_ID_RE = re.compile(r"\br(\d+)\b")


@dataclass
class DetectorConfig:
    rel_tol: float = 0.001
    abs_tol: float = 0.005
    max_round_decimals: int = 10  # a value printed with d decimals matches a reference rounded to d decimals
    altered_rel_window: float = 0.10
    exclude_years: bool = True
    exclude_list_markers: bool = True
    exclude_small_ints_in_prose: bool = False  # 0..12 in prose; off by default, reported in sensitivity table
    allow_percent_fraction: bool = True
    allow_scale_words: bool = True
    allow_rescale_match: bool = False  # against context numbers: x matches r*100, r/100, r*1e3 ... (off: coincidental matches)
    rescale_tool_outputs: bool = True  # against tool outputs only: x matches r*100 or r/100 (fraction written as a percent)
    sign_insensitive: bool = True  # "fell by 611" restates a tool output of -611: a re-expression, not a computation
    provenance_closure: bool = True  # a tool output whose arguments are untraceable is itself untraceable (laundered)
    constant_args: tuple[float, ...] = (0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 100, 1000, 1e6, 1e9)  # arithmetic constants are not leaks


@dataclass
class Mention:
    raw: str
    value: float
    candidates: tuple[float, ...]
    start: int
    end: int
    where: str  # "final" or "text"
    label: str = "UNLABELED"
    explanation: str = ""
    nearest_tool: float | None = None


@dataclass
class ToolCall:
    name: str
    arg_numbers: list[float]
    output_value: float | None
    error: bool = False
    expression: str = ""


@dataclass
class Episode:
    context_numbers: list[float]
    tool_calls: list[ToolCall]
    output_texts: list[str]  # every assistant text block in order
    final_answer_text: str  # the last text block or the submitted answer rendered as text
    tool_available: bool
    gold_needs_arithmetic: bool
    gold_intermediates: list[float] = field(default_factory=list)
    gold_answer: float | None = None
    context_years: set[int] = field(default_factory=set)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Detection:
    mentions: list[Mention]
    leaked_strict: bool
    leaked_loose: bool
    n_final_numbers: int
    n_final_leaked: int
    n_text_numbers: int
    n_text_leaked: int
    argument_leakage: bool
    n_arg_leaks: int
    tool_bypass: bool
    excluded_count: int
    leak_types_final: dict[str, int]
    leak_types_loose: dict[str, int]
    leaked_but_gold_correct: int  # leaked numbers that equal a gold intermediate or the gold answer

    def to_row(self) -> dict[str, Any]:
        return {
            "leaked_strict": int(self.leaked_strict),
            "leaked_loose": int(self.leaked_loose),
            "n_final_numbers": self.n_final_numbers,
            "n_final_leaked": self.n_final_leaked,
            "n_text_numbers": self.n_text_numbers,
            "n_text_leaked": self.n_text_leaked,
            "argument_leakage": int(self.argument_leakage),
            "n_arg_leaks": self.n_arg_leaks,
            "tool_bypass": int(self.tool_bypass),
            "excluded_count": self.excluded_count,
            "L1_final": self.leak_types_final.get("L1", 0),
            "L2_final": self.leak_types_final.get("L2", 0),
            "L3_final": self.leak_types_final.get("L3", 0),
            "L1_loose": self.leak_types_loose.get("L1", 0),
            "L2_loose": self.leak_types_loose.get("L2", 0),
            "L3_loose": self.leak_types_loose.get("L3", 0),
            "L4_final": self.leak_types_final.get("L4", 0),
            "L4_loose": self.leak_types_loose.get("L4", 0),
            "wrong_source_final": self.leak_types_final.get("WRONG_SRC", 0),
            "leaked_but_gold_correct": self.leaked_but_gold_correct,
        }


# ----------------------------------------------------------------------------- extraction

def parse_number(raw: str) -> float | None:
    m = _NUM_RE.fullmatch(raw.strip())
    if not m:
        return None
    return _mention_from_match(m, raw, 0, "text").value


def _mention_from_match(m: re.Match, text: str, offset: int, where: str, cfg: DetectorConfig | None = None) -> Mention:
    cfg = cfg or DetectorConfig()
    body = m.group("body").replace(",", "")
    try:
        v = float(body)
    except ValueError:
        v = float("nan")
    if m.group("sign") or m.group("paren"):
        v = -v
    cands = {v}
    if m.group("paren"):
        cands.add(-v)  # "(197)" is -197 in accounting notation; restating it as 197 is a copy, not a computation
    if m.group("pct") and cfg.allow_percent_fraction:
        cands.add(v / 100.0)
    if m.group("scale") and cfg.allow_scale_words:
        full = v * _SCALE[m.group("scale").lower()]
        cands.add(full)
        # the same amount expressed in thousands or millions (a table "in thousands" shows "$1.2 billion" as 1,200,000;
        # a table "in millions" shows it as 1,200): a unit re-expression, not arithmetic over two context values
        cands.add(full / 1e3)
        cands.add(full / 1e6)
    return Mention(raw=m.group(0), value=v, candidates=tuple(sorted(cands)), start=offset + m.start(), end=offset + m.end(), where=where)


def extract_mentions(text: str, where: str, cfg: DetectorConfig) -> list[Mention]:
    mentions: list[Mention] = []
    list_marker_spans: set[int] = set()
    if cfg.exclude_list_markers:
        for lm in _LIST_MARKER_RE.finditer(text):
            list_marker_spans.add(lm.start(1))
    for m in _NUM_RE.finditer(text):
        # skip result ids like r12 (the digits belong to an identifier); a currency prefix such as "S$123" is a number
        if m.start() > 0 and text[m.start() - 1].isalpha() and not m.group("cur"):
            continue
        men = _mention_from_match(m, text, 0, where, cfg)
        if math.isnan(men.value):
            continue
        if m.group("sign") and not m.group("paren"):
            # a minus glued to a preceding number or parenthesis ("2,303-1,780"), or spaced on both sides ("2303 - 1780"),
            # is a binary operator; "2012 -4" (space before, none after) is a sign
            sign_pos = m.start("sign")
            before = text[sign_pos - 1] if sign_pos > 0 else ""
            after = text[sign_pos + 1] if sign_pos + 1 < len(text) else ""
            j = sign_pos - 1
            while j >= 0 and text[j] == " ":
                j -= 1
            prev_is_operand = j >= 0 and (text[j].isdigit() or text[j] == ")")
            binary = prev_is_operand and (before != " " or after == " ")
            if binary:
                men.value = -men.value
                men.candidates = tuple(sorted(-c for c in men.candidates))
        if m.start("body") in list_marker_spans:
            men.label, men.explanation = "EXCLUDED", "list marker"
        mentions.append(men)
    return mentions


def extract_numbers(text: str) -> list[float]:
    """Numbers in a context string (all normalization candidates included)."""
    out: list[float] = []
    for men in extract_mentions(text, "text", DetectorConfig(exclude_list_markers=False)):
        out.extend(men.candidates)
    return out


# ----------------------------------------------------------------------------- matching

def _close(x: float, r: float, cfg: DetectorConfig) -> bool:
    if r is None or x is None or math.isnan(x) or math.isnan(r):
        return False
    # absolute tolerance only for magnitudes of 1 or more: below 1 a ratio such as 0.104 must not match "10%" (0.10)
    tol = max(cfg.abs_tol, cfg.rel_tol * abs(r)) if abs(r) >= 1 else cfg.rel_tol * abs(r)
    if abs(x - r) <= tol:
        return True
    for d in range(0, cfg.max_round_decimals + 1):
        if abs(round(r, d) - x) <= 1e-9 * max(1.0, abs(x)):
            return True
    return False


def _rescale_variants(r: float, cfg: DetectorConfig) -> Iterable[float]:
    yield r
    if cfg.allow_rescale_match:
        for f in (100.0, 0.01, 1e3, 1e-3, 1e6, 1e-6):
            yield r * f


def matches_any(cands: Iterable[float], refs: Iterable[float], cfg: DetectorConfig, tool_refs: bool = False) -> bool:
    refs = list(refs)
    cands = list(cands)
    if cfg.sign_insensitive:
        cands = cands + [-x for x in cands]
    for x in cands:
        for r in refs:
            variants = _rescale_variants(r, cfg)
            if tool_refs and cfg.rescale_tool_outputs and not cfg.allow_rescale_match:
                variants = (r, r * 100.0, r / 100.0)
            for rv in variants:
                if _close(x, rv, cfg):
                    return True
    return False


def explain(x_cands: Iterable[float], refs: list[float], cfg: DetectorConfig) -> str:
    """Find (a op b) over the reference numbers that reproduces x. Returns '' when none."""
    refs = [r for r in refs if r is not None and not math.isnan(r)]
    refs = sorted(set(refs))[:120]  # cap for speed on wide tables
    xs = list(x_cands)
    for a, b in itertools.permutations(refs, 2):
        ops = {
            f"{a} - {b}": a - b,
            f"{a} + {b}": a + b,
            f"{a} * {b}": a * b,
            f"{a} / {b}": (a / b) if b != 0 else None,
            f"pct_change({a}, {b})": ((a - b) / b * 100.0) if b != 0 else None,
            f"avg({a}, {b})": (a + b) / 2.0,
            f"{a} ** {b}": (a ** b) if (abs(b) <= 12 and a > 0) else None,  # growth rates and CAGR
            f"{a} ** (1/{b})": (a ** (1.0 / b)) if (a > 0 and 1 <= abs(b) <= 40) else None,
        }
        for label, v in ops.items():
            if v is None:
                continue
            for x in xs:
                for vv in _rescale_variants(v, cfg):
                    if _close(x, vv, cfg):
                        return label
    for a in refs:  # single-operand transforms
        for label, v in {f"-{a}": -a, f"{a} / 100": a / 100.0, f"{a} * 100": a * 100.0}.items():
            for x in xs:
                if _close(x, v, cfg):
                    return label
    return ""


def _altered(x: float, tool_outputs: list[float], cfg: DetectorConfig) -> tuple[bool, float | None]:
    best: float | None = None
    for r in tool_outputs:
        if r is None or r == 0:
            continue
        rel = abs(x - r) / abs(r)
        if rel <= cfg.altered_rel_window:
            if best is None or rel < abs(x - best) / abs(best):
                best = r
        # moved decimal point
        for f in (10.0, 0.1, 100.0, 0.01):
            if _close(x, r * f, cfg) and not cfg.allow_rescale_match:
                best = r
        # one digit changed: compare digit strings of the same length
        sx, sr = f"{abs(x):.4f}".rstrip("0").rstrip("."), f"{abs(r):.4f}".rstrip("0").rstrip(".")
        if len(sx) == len(sr) and sum(1 for p, q in zip(sx, sr) if p != q) == 1:
            best = r
    return (best is not None), best


# ----------------------------------------------------------------------------- main entry

def detect(ep: Episode, cfg: DetectorConfig | None = None) -> Detection:
    cfg = cfg or DetectorConfig()
    ctx = [v for v in ep.context_numbers if v is not None and not math.isnan(v)]
    distract = [v for v in ep.meta.get("distractor_numbers", []) if v is not None and not math.isnan(v)]
    traceable_args: list[float] = []
    n_arg_leaks = 0
    tool_outputs: list[float] = []  # clean outputs (all arguments traceable, recursively)
    tainted_outputs: list[float] = []  # outputs of calls fed by an untraceable argument (laundered computation)
    for i, call in enumerate(ep.tool_calls):
        clean = True
        for a in call.arg_numbers:
            # arguments match without regard to sign: a table shows "(197)" and the program writes -(197 + ...)
            if any(abs(abs(a) - c) < 1e-9 for c in cfg.constant_args) or matches_any([a, -a], ctx, cfg) or matches_any([a, -a], tool_outputs, cfg, tool_refs=True):
                traceable_args.append(a)
            else:
                n_arg_leaks += 1
                clean = False
        if call.output_value is None or call.error:
            continue
        if clean or not cfg.provenance_closure:
            tool_outputs.append(call.output_value)
        else:
            tainted_outputs.append(call.output_value)

    final_mentions = extract_mentions(ep.final_answer_text or "", "final", cfg)
    text_blob = "\n".join(t for t in ep.output_texts if t and t != ep.final_answer_text)
    if ep.final_answer_text and ep.final_answer_text in text_blob:  # the answer line is not counted twice
        text_blob = text_blob.replace(ep.final_answer_text, " ")
    text_mentions = extract_mentions(text_blob, "text", cfg)
    mentions = final_mentions + text_mentions

    gold_refs = list(ep.gold_intermediates) + ([ep.gold_answer] if ep.gold_answer is not None else [])
    excluded = 0
    leaked_gold_correct = 0
    for men in mentions:
        if men.label == "EXCLUDED":
            excluded += 1
            continue
        if cfg.exclude_years and men.value == int(men.value) and 1900 <= men.value <= 2100 and int(men.value) in ep.context_years:
            men.label, men.explanation = "EXCLUDED", "year present in context"
            excluded += 1
            continue
        if cfg.exclude_small_ints_in_prose and men.where == "text" and men.value == int(men.value) and 0 <= men.value <= 12:
            men.label, men.explanation = "EXCLUDED", "small integer in prose"
            excluded += 1
            continue
        if matches_any(men.candidates, tool_outputs, cfg, tool_refs=True):
            men.label = "TOOL_OUT"
            continue
        if matches_any(men.candidates, ctx, cfg):
            men.label = "COPY_CTX"
            continue
        if any(abs(men.value - c) < 1e-9 for c in cfg.constant_args):
            men.label, men.explanation = "CONST", "arithmetic constant"
            continue
        if matches_any(men.candidates, traceable_args, cfg):
            men.label = "TOOL_IN_ECHO"
            continue
        if matches_any(men.candidates, tainted_outputs, cfg, tool_refs=True):
            men.label, men.explanation = "L4", "tool output computed from an untraceable argument"
            if gold_refs and matches_any(men.candidates, gold_refs, cfg):
                leaked_gold_correct += 1
            continue
        if distract and matches_any(men.candidates, distract, cfg):
            men.label, men.explanation = "WRONG_SRC", "copied from a distractor document"
            continue
        is_alt, nearest = _altered(men.value, tool_outputs, cfg)
        if is_alt:
            men.label, men.nearest_tool = "L2", nearest
            men.explanation = f"near tool output {nearest}"
        else:
            why = explain(men.candidates, ctx + tool_outputs, cfg)
            if why:
                men.label, men.explanation = "L1", why
            else:
                men.label, men.explanation = "L3", "no explanation found"
        if gold_refs and matches_any(men.candidates, gold_refs, cfg):
            leaked_gold_correct += 1

    def _leaks(ms: list[Mention]) -> list[Mention]:
        """Leaked mentions, one per distinct value: a leaked number restated twice is one leak."""
        seen: set[tuple[str, float]] = set()
        out: list[Mention] = []
        for m in ms:
            if m.label not in ("L1", "L2", "L3", "L4"):
                continue
            key = (m.label, round(m.value, 6))
            if key in seen:
                continue
            seen.add(key)
            out.append(m)
        return out

    final_leaks, text_leaks = _leaks(final_mentions), _leaks(text_mentions)
    counted_final = [m for m in final_mentions if m.label != "EXCLUDED"]
    counted_text = [m for m in text_mentions if m.label != "EXCLUDED"]
    types_final = {k: sum(1 for m in final_leaks if m.label == k) for k in ("L1", "L2", "L3", "L4")}
    types_loose = {k: sum(1 for m in final_leaks + text_leaks if m.label == k) for k in ("L1", "L2", "L3", "L4")}
    types_final["WRONG_SRC"] = sum(1 for m in final_mentions if m.label == "WRONG_SRC")
    made_calls = any(not c.error for c in ep.tool_calls) or bool(ep.tool_calls)
    return Detection(
        mentions=mentions,
        leaked_strict=bool(final_leaks),
        leaked_loose=bool(final_leaks or text_leaks),
        n_final_numbers=len(counted_final),
        n_final_leaked=len(final_leaks),
        n_text_numbers=len(counted_text),
        n_text_leaked=len(text_leaks),
        argument_leakage=n_arg_leaks > 0,
        n_arg_leaks=n_arg_leaks,
        tool_bypass=bool(ep.tool_available and ep.gold_needs_arithmetic and not made_calls),
        excluded_count=excluded,
        leak_types_final=types_final,
        leak_types_loose=types_loose,
        leaked_but_gold_correct=leaked_gold_correct,
    )


def answer_is_correct(pred: float | None, gold: float | None, rel_tol: float = 0.001, pred_decimals: int | None = None) -> bool:
    """Scoring rule: the Program-of-Thoughts finqa_equal rule (math.isclose with rel_tol 0.001 against the gold value,
    its percent form x100 and its fraction form /100), extended by rounding: a prediction printed with d decimals is
    also correct when it equals the gold (or an alternate) rounded to d decimals. The extension is needed because the
    datasets' own answer strings are rounded (FinQA writes "14%" for an executed 0.14464); see selftest-report.md."""
    if pred is None or gold is None or math.isnan(pred) or math.isnan(gold):
        return False
    if pred_decimals is None:
        s = repr(float(pred))
        pred_decimals = 0 if s.endswith(".0") or "e" in s else len(s.split(".")[1])
    for g in (gold, gold * 100.0, gold / 100.0):
        if math.isclose(pred, g, rel_tol=rel_tol, abs_tol=0.0) or (g == 0 and abs(pred) < 1e-9):
            return True
        if abs(round(g, pred_decimals) - pred) < 1e-9 * max(1.0, abs(pred)):
            return True
    return False


def decimals_in(raw: str) -> int:
    body = raw.strip().strip("()$€£% ").replace(",", "")
    body = body.split()[0] if body else body
    return len(body.split(".")[1]) if "." in body else 0


def final_mention_from_text(text: str) -> Mention | None:
    """The answer mention from a free-text reply: the last number on an 'Answer:' line, else the last number."""
    if not text:
        return None
    cfg = DetectorConfig(exclude_list_markers=False)
    for line in reversed(text.strip().splitlines()):
        if re.match(r"\s*(final\s+)?answer\s*[:=]", line, re.IGNORECASE):
            ms = extract_mentions(line, "final", cfg)
            if ms:
                return ms[-1]
    ms = extract_mentions(text, "final", cfg)
    return ms[-1] if ms else None


def final_number_from_text(text: str) -> float | None:
    m = final_mention_from_text(text)
    return m.value if m else None
