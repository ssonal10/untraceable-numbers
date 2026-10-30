"""Pluggable LLM clients: a scripted mock (for dry runs and detector validation), the
official Anthropic SDK client (default for real runs), and an OpenAI-compatible client
for open-weights models served through a compatible endpoint.

All clients expose one method:
    complete(system, messages, tools, *, max_tokens, effort, temperature, json_schema) -> LLMResponse
where `messages` use the Anthropic Messages API block shapes
({"role": "user"|"assistant", "content": str | [blocks]}) and tools use the Anthropic
tool schema. The OpenAI-compatible client converts both.

Spending guard: real clients refuse to make more than P2_MAX_CALLS calls per process
(default 50). The dry run never constructs a real client.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any

from calculator import format_number


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class LLMResponse:
    blocks: list[dict[str, Any]]
    stop_reason: str
    usage: dict[str, int]
    model: str

    @property
    def text_blocks(self) -> list[str]:
        return [b["text"] for b in self.blocks if b.get("type") == "text"]

    @property
    def tool_uses(self) -> list[dict[str, Any]]:
        return [b for b in self.blocks if b.get("type") == "tool_use"]


def _approx_tokens(obj: Any) -> int:
    return max(1, len(json.dumps(obj, default=str)) // 4)


# ----------------------------------------------------------------------------- mock

@dataclass
class MockPolicy:
    """How the scripted model behaves. Probabilities are per episode or per step as noted."""

    leak_p: float = 0.15  # per step: compute in text instead of calling the tool
    brevity_leak_p: float = 0.45  # per step under BREVITY pressure
    budget_extra_leak_p: float = 0.9  # per step after the first under BUDGET pressure
    failure_fallback_p: float = 0.6  # per episode: after a failed call, compute everything in text
    alter_p: float = 0.05  # per episode: restate the final tool output with a changed digit
    bypass_p: float = 0.05  # per episode: never call the tool (OPTIONAL regime)
    wrong_text_math_p: float = 0.3  # per text-computed step: make an arithmetic error
    seed: int = 0


@dataclass
class MockClient:
    """Scripted policy over the item's gold steps. Records ground truth for every emitted number."""

    policy: MockPolicy = field(default_factory=MockPolicy)
    model_name: str = "mock-model"
    _rng: random.Random = field(default_factory=random.Random)
    _item: Any = None
    _cond: Any = None
    _plan: list[str] = field(default_factory=list)  # per step: "tool" or "text"
    _text_values: dict[int, float] = field(default_factory=dict)
    truth: dict[str, Any] = field(default_factory=dict)
    calls: int = 0

    def begin_episode(self, item: Any, cond: Any, seed: int) -> None:
        self._rng = random.Random(f"{self.policy.seed}:{seed}:{item.id}:{cond.key()}")
        self._item, self._cond = item, cond
        self.truth = {"truth_text_computed": [], "truth_altered": None, "truth_bypass": False, "truth_fallback": False}
        self._text_values = {}
        n = len(item.gold_steps)
        regime, pressure = cond.regime, cond.pressure
        if regime == "NOTOOL":
            self._plan = ["text"] * n
            return
        p = self.policy
        if regime == "OPTIONAL" and self._rng.random() < p.bypass_p:
            self._plan = ["text"] * n
            self.truth["truth_bypass"] = True
            return
        per_step = p.leak_p
        if pressure == "BREVITY":
            per_step = p.brevity_leak_p
        plan = []
        for i in range(n):
            if pressure == "BUDGET" and i > 0:
                plan.append("text" if self._rng.random() < p.budget_extra_leak_p else "tool")
            else:
                plan.append("text" if self._rng.random() < per_step else "tool")
        self._plan = plan

    # -- helpers
    def _step_value(self, idx: int, tool_values: dict[int, float]) -> float | None:
        """Value of gold step idx as the mock currently knows it (tool result or its own text math)."""
        if idx in tool_values:
            return tool_values[idx]
        return self._text_values.get(idx)

    def _resolve_operand(self, expr_tok: str, tool_ids: dict[int, str], tool_values: dict[int, float]) -> str:
        m = re.fullmatch(r"#(\d+)", expr_tok.strip())
        if not m:
            return expr_tok.strip()
        idx = int(m.group(1))
        if idx in tool_ids:
            return tool_ids[idx]
        v = self._text_values.get(idx)
        if v is not None:
            self.truth["truth_laundered"] = True  # a text-computed value enters the tool: every downstream output is tainted
        return format_number(v) if v is not None else "0"

    def _text_compute(self, idx: int, tool_values: dict[int, float]) -> str:
        step = self._item.gold_steps[idx]
        true_val = step.value
        if true_val is None:
            return ""
        val = true_val
        if self._rng.random() < self.policy.wrong_text_math_p:
            val = true_val * (1 + self._rng.choice([-0.02, 0.015, 0.05])) + self._rng.choice([0, 0.1, -0.1])
        self._text_values[idx] = val
        self.truth["truth_text_computed"].append(val)
        if self._cond.pressure != "BREVITY" and not getattr(self, "_json_mode", False):
            # the step line is emitted in prose, so the number is visible to the detector (not under BREVITY or JSON output)
            self.truth.setdefault("truth_emitted_leaks", []).append(val)
        expr = step.expr
        for ref in re.findall(r"#(\d+)", expr):
            rv = self._step_value(int(ref), tool_values)
            expr = expr.replace(f"#{ref}", format_number(rv) if rv is not None else "?")
        return f"Step {idx + 1}: {expr} = {format_number(val)}."

    def complete(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *, max_tokens: int = 1024,
                 effort: str = "low", temperature: float | None = None, json_schema: dict | None = None) -> LLMResponse:
        self.calls += 1
        item, cond = self._item, self._cond
        steps = item.gold_steps
        # Reconstruct what the tool has returned so far from the conversation.
        tool_ids: dict[int, str] = {}
        tool_values: dict[int, float] = {}
        pending_use_ids: list[str] = []
        failed_once = False
        step_of_use: dict[str, int] = {}
        gate_rejected = False
        for msg in messages:
            if msg["role"] == "assistant" and isinstance(msg["content"], list):
                for b in msg["content"]:
                    if b.get("type") == "tool_use" and b.get("name") == "calculator":
                        step_of_use[b["id"]] = b["input"].get("_step", -1)
            if msg["role"] == "user" and isinstance(msg["content"], list):
                for b in msg["content"]:
                    if b.get("type") == "tool_result":
                        content = b.get("content", "")
                        txt = content if isinstance(content, str) else json.dumps(content)
                        m = re.search(r"(r\d+) = (-?[\d.]+)", txt)
                        idx = step_of_use.get(b["tool_use_id"], -1)
                        if m and idx >= 0:
                            tool_ids[idx], tool_values[idx] = m.group(1), float(m.group(2))
                        elif "ERROR" in txt:
                            failed_once = True
                    elif b.get("type") == "text" and "not from a calculator result" in b.get("text", ""):
                        gate_rejected = True
                    if b.get("type") == "tool_result" and "rejected: value must be" in str(b.get("content", "")):
                        gate_rejected = True  # the answer contract rejected a literal: same reaction as the gate
        if gate_rejected:  # the gate asked for a retry: switch every remaining step to the tool
            self._plan = ["tool"] * len(steps)
            self._text_values = {}
            self.truth["truth_text_computed"] = []
            self.truth["truth_emitted_leaks"] = []  # the rejected attempt is not what the user sees
            self.truth["truth_altered"] = None
            self.truth["truth_laundered"] = False
        self._json_mode = json_schema is not None
        if failed_once and not self.truth["truth_fallback"] and not tool_values:
            if self._rng.random() < self.policy.failure_fallback_p:
                self._plan = ["text"] * len(steps)
                self.truth["truth_fallback"] = True
        blocks: list[dict[str, Any]] = []
        out_text: list[str] = []
        # Walk the steps in order; emit the next tool call when a step is planned for the tool and not yet done.
        for idx, step in enumerate(steps):
            if idx in tool_values or idx in self._text_values:
                continue
            if self._plan[idx] == "text" or not step.expr:
                line = self._text_compute(idx, tool_values)
                if line and cond.pressure != "BREVITY":
                    out_text.append(line)
                continue
            expr = step.expr
            for tok in re.findall(r"#\d+", expr):
                expr = expr.replace(tok, self._resolve_operand(tok, tool_ids, tool_values))
            uid = f"toolu_mock_{hashlib.md5(f'{item.id}{cond.key()}{idx}{self.calls}'.encode()).hexdigest()[:10]}"
            if out_text and cond.pressure != "BREVITY":
                blocks.append({"type": "text", "text": " ".join(out_text)})
            blocks.append({"type": "tool_use", "id": uid, "name": "calculator", "input": {"expression": expr, "_step": idx}})
            return LLMResponse(blocks=blocks, stop_reason="tool_use", usage=self._usage(system, messages, blocks), model=self.model_name)
        # All steps resolved: produce the final answer.
        final_idx = len(steps) - 1
        final_val = self._step_value(final_idx, tool_values) if steps else item.gold_answer
        if final_val is None:
            final_val = item.gold_answer if item.gold_answer is not None else 0.0
        if item.answer_scale == "percent" and final_val is not None and abs(final_val) < 1.0 and item.gold_answer is not None and abs(item.gold_answer) >= 1.0:
            final_val = final_val * 100.0
        altered = False
        if final_idx in tool_values and self._rng.random() < self.policy.alter_p:
            final_val = round(final_val * 1.01 + 0.03, 2)
            altered = True
            self.truth["truth_altered"] = final_val
        if steps and final_idx not in tool_values:  # the final number itself was computed in text and is emitted
            self.truth.setdefault("truth_emitted_leaks", []).append(final_val)
        if cond.regime == "CONTRACT":
            value = tool_ids.get(final_idx) if (final_idx in tool_ids and not altered) else format_number(final_val)
            uid = f"toolu_mock_submit_{self.calls}"
            blocks.append({"type": "tool_use", "id": uid, "name": "submit_answer",
                           "input": {"value": value, "unit": item.answer_scale or "none", "derivation": ", ".join(tool_ids.values())}})
            return LLMResponse(blocks=blocks, stop_reason="tool_use", usage=self._usage(system, messages, blocks), model=self.model_name)
        if json_schema is not None:
            payload = {"value": final_val, "unit": item.answer_scale or "none",
                       "source_ids": [tool_ids[final_idx]] if final_idx in tool_ids and not altered else ["context"]}
            blocks.append({"type": "text", "text": json.dumps(payload)})
        else:
            prose = " ".join(out_text) + ("\n" if out_text else "")
            if cond.pressure == "BREVITY":
                prose = ""
            blocks.append({"type": "text", "text": f"{prose}Answer: {format_number(final_val)}"})
        return LLMResponse(blocks=blocks, stop_reason="end_turn", usage=self._usage(system, messages, blocks), model=self.model_name)

    def _usage(self, system: str, messages: list[dict[str, Any]], blocks: list[dict[str, Any]]) -> dict[str, int]:
        return {"input_tokens": _approx_tokens(system) + _approx_tokens(messages), "output_tokens": _approx_tokens(blocks) + 120}


# ----------------------------------------------------------------------------- Anthropic (official SDK)

class AnthropicClient:
    """Official SDK client. Model id from the argument or P2_MODEL (default claude-opus-5-5).

    No fallback model is configured on purpose: a refusal is recorded as stop_reason
    'refusal' so the model factor of the experiment stays clean.
    """

    def __init__(self, model: str | None = None, max_calls: int | None = None) -> None:
        import anthropic  # lazy: the dry run must work without the SDK installed

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model_name = model or os.environ.get("P2_MODEL", "claude-opus-5-5")
        self.max_calls = max_calls if max_calls is not None else int(os.environ.get("P2_MAX_CALLS", "50"))
        self.calls = 0
        self.truth: dict[str, Any] = {}

    def begin_episode(self, item: Any, cond: Any, seed: int) -> None:  # parity with the mock
        self.truth = {}

    def complete(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *, max_tokens: int = 4096,
                 effort: str = "low", temperature: float | None = None, json_schema: dict | None = None) -> LLMResponse:
        if self.calls >= self.max_calls:
            raise BudgetExceeded(f"P2_MAX_CALLS={self.max_calls} reached; raise it deliberately to spend more")
        self.calls += 1
        clean_msgs = _strip_private_keys(messages)
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "system": system,
            "messages": clean_msgs,
            "output_config": {"effort": effort},
        }
        if tools:
            kwargs["tools"] = tools
        if json_schema is not None:
            kwargs["output_config"]["format"] = {"type": "json_schema", "schema": json_schema}
        if temperature is not None:  # only for models that accept it; current 5.5 models return 400
            kwargs["temperature"] = temperature
        a = self._anthropic
        for attempt in range(5):
            try:
                resp = self.client.messages.create(**kwargs)
                break
            except a.RateLimitError as e:
                time.sleep(int(e.response.headers.get("retry-after", "20")))
            except a.APIStatusError as e:
                if e.status_code >= 500 and attempt < 4:
                    time.sleep(5 * (attempt + 1))
                    continue
                raise
            except a.APIConnectionError:
                if attempt < 4:
                    time.sleep(5 * (attempt + 1))
                    continue
                raise
        else:
            raise RuntimeError("gave up after retries")
        blocks: list[dict[str, Any]] = []
        for b in resp.content:
            if b.type == "text":
                blocks.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": dict(b.input)})
            elif b.type == "thinking":
                blocks.append({"type": "thinking", "thinking": getattr(b, "thinking", ""), "signature": getattr(b, "signature", None)})
        usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens,
                 "cache_read_input_tokens": getattr(resp.usage, "cache_read_input_tokens", 0) or 0}
        return LLMResponse(blocks=blocks, stop_reason=resp.stop_reason, usage=usage, model=resp.model)


# ----------------------------------------------------------------------------- OpenAI-compatible (open-weights models only)

class OpenAICompatClient:
    """For open-weights models behind an OpenAI-compatible endpoint (vLLM, hosted providers).
    Not used for Claude. Untested in this study (no endpoint available); kept minimal."""

    def __init__(self, model: str | None = None, base_url: str | None = None, max_calls: int | None = None) -> None:
        import openai  # lazy

        self.client = openai.OpenAI(base_url=base_url or os.environ.get("P2_OPENAI_BASE_URL"), api_key=os.environ.get("P2_OPENAI_API_KEY", "none"))
        self.model_name = model or os.environ.get("P2_OPENAI_MODEL", "")
        self.max_calls = max_calls if max_calls is not None else int(os.environ.get("P2_MAX_CALLS", "50"))
        self.calls = 0
        self.truth: dict[str, Any] = {}

    def begin_episode(self, item: Any, cond: Any, seed: int) -> None:
        self.truth = {}

    def complete(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *, max_tokens: int = 4096,
                 effort: str = "low", temperature: float | None = None, json_schema: dict | None = None) -> LLMResponse:
        if self.calls >= self.max_calls:
            raise BudgetExceeded(f"P2_MAX_CALLS={self.max_calls} reached")
        self.calls += 1
        oai_msgs: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for m in _strip_private_keys(messages):
            if isinstance(m["content"], str):
                oai_msgs.append({"role": m["role"], "content": m["content"]})
                continue
            if m["role"] == "assistant":
                text = " ".join(b["text"] for b in m["content"] if b.get("type") == "text")
                calls = [{"id": b["id"], "type": "function", "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}
                         for b in m["content"] if b.get("type") == "tool_use"]
                entry: dict[str, Any] = {"role": "assistant", "content": text or None}
                if calls:
                    entry["tool_calls"] = calls
                oai_msgs.append(entry)
            else:
                for b in m["content"]:
                    if b.get("type") == "tool_result":
                        c = b.get("content", "")
                        oai_msgs.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": c if isinstance(c, str) else json.dumps(c)})
                    elif b.get("type") == "text":
                        oai_msgs.append({"role": "user", "content": b["text"]})
        kwargs: dict[str, Any] = {"model": self.model_name, "messages": oai_msgs, "max_tokens": max_tokens}
        if tools:
            kwargs["tools"] = [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}} for t in tools]
        if temperature is not None:
            kwargs["temperature"] = temperature
        if json_schema is not None:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {"name": "answer", "schema": json_schema}}
        resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        blocks: list[dict[str, Any]] = []
        if choice.message.content:
            blocks.append({"type": "text", "text": choice.message.content})
        for tc in choice.message.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {"expression": tc.function.arguments}
            blocks.append({"type": "tool_use", "id": tc.id, "name": tc.function.name, "input": args})
        stop = "tool_use" if choice.message.tool_calls else "end_turn"
        usage = {"input_tokens": resp.usage.prompt_tokens, "output_tokens": resp.usage.completion_tokens}
        return LLMResponse(blocks=blocks, stop_reason=stop, usage=usage, model=resp.model)


def _strip_private_keys(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove harness-private keys (such as the mock's _step marker) before sending to an API."""
    out = []
    for m in messages:
        if isinstance(m["content"], list):
            blocks = []
            for b in m["content"]:
                if b.get("type") == "tool_use":
                    b = {**b, "input": {k: v for k, v in b["input"].items() if not k.startswith("_")}}
                if b.get("type") == "thinking":
                    pass  # thinking blocks are passed back unchanged
                blocks.append(b)
            out.append({"role": m["role"], "content": blocks})
        else:
            out.append(m)
    return out


def make_client(kind: str, model: str | None = None, policy: MockPolicy | None = None) -> Any:
    if kind == "mock":
        return MockClient(policy=policy or MockPolicy())
    if kind == "anthropic":
        return AnthropicClient(model=model)
    if kind == "openai-compat":
        return OpenAICompatClient(model=model)
    raise ValueError(f"unknown client kind {kind!r}")
