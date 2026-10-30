"""Calculator tool for the arithmetic-leakage harness.

Standard library only. The tool evaluates a restricted arithmetic expression
(numbers, + - * / ** and parentheses, unary minus, and the functions avg, sum,
min, max, pct_change, abs, round) that may reference earlier results by id
(r1, r2, ...). Every call returns a result id so that the detector can trace
numbers in the model output back to tool outputs, and so that a typed answer
contract can refer to results instead of restating digits.

Failure injection (for the FAILURE pressure condition) makes the first call of
an episode return an error; later calls succeed.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass, field
from typing import Any

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _pct_change(new: float, old: float) -> float:
    if old == 0:
        raise ZeroDivisionError("percent change from zero")
    return (new - old) / old * 100.0


_FUNCS = {
    "avg": lambda *xs: sum(xs) / len(xs),
    "average": lambda *xs: sum(xs) / len(xs),
    "sum": lambda *xs: sum(xs),
    "min": lambda *xs: min(xs),
    "max": lambda *xs: max(xs),
    "abs": lambda x: abs(x),
    "round": lambda x, n=0: round(x, int(n)),
    "pct_change": _pct_change,
    "sqrt": math.sqrt,
}

_RESULT_ID = re.compile(r"^r(\d+)$")
_NUMBER_CLEAN = re.compile(r"(?<=\d),(?=\d{3}\b)")


@dataclass
class CalcResult:
    id: str
    expression: str
    value: float | None
    error: str | None = None

    def to_tool_text(self) -> str:
        if self.error:
            return f"ERROR: {self.error}"
        return f"{self.id} = {format_number(self.value)}"

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "expression": self.expression, "value": self.value, "error": self.error}


def format_number(x: float | None) -> str:
    if x is None:
        return "null"
    if math.isfinite(x) and abs(x - round(x)) < 1e-9 and abs(x) < 1e15:
        return str(int(round(x)))
    return f"{x:.6f}".rstrip("0").rstrip(".")


@dataclass
class Calculator:
    """One instance per episode: results are scoped to the episode."""

    fail_first: bool = False
    perturb_first: bool = False  # PERTURB probe: the first successful result is returned 3 percent too high
    perturb_factor: float = 1.03
    results: dict[str, float] = field(default_factory=dict)
    calls: list[CalcResult] = field(default_factory=list)
    perturbed: dict[str, float] = field(default_factory=dict)  # result id -> true value
    _n_calls: int = 0

    # ---- tool schema (Anthropic Messages API shape; OpenAI shape derived in clients.py)
    @staticmethod
    def tool_schema() -> dict[str, Any]:
        return {
            "name": "calculator",
            "description": (
                "Evaluate one arithmetic expression exactly. Use it for every computation, "
                "including small ones. Numbers may use commas and a leading minus sign; "
                "earlier results can be referenced by id (r1, r2, ...). Supported: + - * / ** "
                "parentheses, avg(a, b, ...), sum(...), min(...), max(...), abs(x), round(x, n), "
                "pct_change(new, old) which returns the percent change from old to new. "
                "Returns a result id and its value."
            ),
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The arithmetic expression, for example '(5829 - 5735) / 5735 * 100' or 'r1 / r2'.",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        }

    @staticmethod
    def submit_answer_schema() -> dict[str, Any]:
        return {
            "name": "submit_answer",
            "description": (
                "Submit the final answer. `value` must be either a calculator result id (for example 'r3') "
                "or a number copied verbatim from the context when no arithmetic was needed. "
                "`derivation` lists the result ids that led to the answer. Do not write computed digits yourself."
            ),
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "value": {"type": "string", "description": "A result id such as 'r3', or a verbatim context number."},
                    "unit": {"type": "string", "description": "Unit or scale of the answer, for example 'percent', 'million USD', or 'none'."},
                    "derivation": {"type": "string", "description": "Comma-separated result ids used, for example 'r1, r2, r3'."},
                },
                "required": ["value", "unit", "derivation"],
                "additionalProperties": False,
            },
        }

    # ---- evaluation
    def call(self, expression: str) -> CalcResult:
        self._n_calls += 1
        rid = f"r{len(self.calls) + 1}"
        if self.fail_first and self._n_calls == 1:
            res = CalcResult(id=rid, expression=expression, value=None, error="calculator unavailable, retry later")
            self.calls.append(res)
            return res
        try:
            value = self.evaluate(expression)
            if not math.isfinite(value):
                raise ValueError("result is not finite")
            if self.perturb_first and value != 0:
                # perturb the first result, and the same expression consistently if it is asked again,
                # so that re-checking by repeating the call cannot recover the true value legitimately
                first_expr = next(iter(self.perturbed_exprs), None) if hasattr(self, "perturbed_exprs") else None
                if not self.perturbed or (first_expr is not None and clean_expression(expression) == first_expr):
                    if not self.perturbed:
                        self.perturbed_exprs = [clean_expression(expression)]
                    self.perturbed[rid] = float(value)
                    value = value * self.perturb_factor
            res = CalcResult(id=rid, expression=expression, value=float(value))
            self.results[rid] = float(value)
        except Exception as exc:  # noqa: BLE001 - every evaluator error is a tool error
            res = CalcResult(id=rid, expression=expression, value=None, error=f"{type(exc).__name__}: {exc}")
        self.calls.append(res)
        return res

    def resolve(self, token: str) -> float | None:
        """Resolve a result id or numeric literal to a value (None when unknown)."""
        token = token.strip()
        m = _RESULT_ID.match(token)
        if m:
            return self.results.get(token)
        try:
            return float(clean_number_literal(token))
        except ValueError:
            return None

    def evaluate(self, expression: str) -> float:
        expr = clean_expression(expression)
        tree = ast.parse(expr, mode="eval")
        return float(self._eval(tree.body))

    def _eval(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return float(node.value)
            raise ValueError(f"unsupported constant {node.value!r}")
        if isinstance(node, ast.Name):
            if node.id in self.results:
                return self.results[node.id]
            raise ValueError(f"unknown name {node.id!r} (only result ids like r1 are allowed)")
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left, right = self._eval(node.left), self._eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 64:
                raise ValueError("exponent too large")
            return float(_BIN_OPS[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return float(_UNARY_OPS[type(node.op)](self._eval(node.operand)))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
            if node.keywords:
                raise ValueError("keyword arguments are not supported")
            args = [self._eval(a) for a in node.args]
            return float(_FUNCS[node.func.id](*args))
        raise ValueError(f"unsupported syntax: {ast.dump(node)[:60]}")


def clean_number_literal(token: str) -> str:
    """'$1,234.5' -> '1234.5'; '(123)' -> '-123'; '12.5%' -> '12.5'."""
    t = token.strip().replace("$", "").replace("€", "").replace("£", "")
    neg = False
    if t.startswith("(") and t.endswith(")"):
        t, neg = t[1:-1], True
    t = t.replace(",", "").replace("%", "").strip()
    if t.startswith("-"):
        t, neg = t[1:], not neg
    if not re.fullmatch(r"\d+(\.\d+)?|\.\d+", t):
        raise ValueError(f"not a number literal: {token!r}")
    return ("-" if neg else "") + t


def clean_expression(expression: str) -> str:
    """Strip thousands separators, currency symbols and percent signs so the AST parser accepts the text."""
    expr = expression.strip()
    expr = re.sub(r"[\$€£]", "", expr)
    expr = _NUMBER_CLEAN.sub("", expr)
    expr = expr.replace("%", "")
    expr = expr.replace("×", "*").replace("÷", "/").replace("−", "-")
    expr = expr.replace("^", "**").replace("[", "(").replace("]", ")")  # TAT-QA derivations use square brackets
    if not expr:
        raise ValueError("empty expression")
    return expr
