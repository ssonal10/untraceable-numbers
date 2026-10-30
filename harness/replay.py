#!/usr/bin/env python3
"""Re-label episodes from stored traces with the current detector and scoring rule (no API calls).

  python3 -I replay.py traces.jsonl [traces2.jsonl ...] --out relabeled.csv

Each trace holds the assistant blocks of every turn; a replay client returns them in order and the episode runner
re-executes the calculator (deterministic, including failure and perturbation injection), re-applies the gate logic
and recomputes every row field. Rows whose replay diverges from the stored trace (for example a gate that would now
have rejected a reply) are marked `replay_diverged = 1` and keep the stored final text.

Items are reloaded from the public files, so P2_DATA_CACHE must point at the cache used for the run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import Condition, EpisodeRunner, RunConfig  # noqa: E402
from clients import LLMResponse  # noqa: E402
from loaders import counterfactual, load_dataset  # noqa: E402


class ReplayClient:
    def __init__(self, trace: dict[str, Any]) -> None:
        self.turns = [m["content"] for m in trace["messages"] if m["role"] == "assistant"]
        self.i = 0
        self.model_name = trace.get("model", "?")
        self.model_reported = trace.get("model_reported", "")
        self.truth: dict[str, Any] = {}

    def begin_episode(self, item: Any, cond: Any, seed: int) -> None:
        return None

    def complete(self, system, messages, tools, **kw) -> LLMResponse:
        if self.i >= len(self.turns):
            raise StopIteration
        blocks = self.turns[self.i]
        self.i += 1
        has_tool = any(b.get("type") == "tool_use" for b in blocks)
        return LLMResponse(blocks=blocks, stop_reason="tool_use" if has_tool else "end_turn", usage={}, model=self.model_reported)


def parse_cond(key: str) -> Condition:
    parts = key.split("|")
    temp = None
    if len(parts) > 4 and parts[4].startswith("t"):
        temp = float(parts[4][1:])
    return Condition(parts[0], parts[1], parts[2], parts[3], temp)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None, help='JSON overrides for DetectorConfig, e.g. \'{"rel_tol": 0.01, "sign_insensitive": false}\'')
    args = ap.parse_args()
    cache: dict[str, dict[str, Any]] = {}

    def get_item(item_id: str):
        ds = item_id.split(":")[0]
        base_id = item_id[:-3] if item_id.endswith(":cf") else item_id
        if ds not in cache:
            cache[ds] = {it.id: it for it in load_dataset(ds)}
        it = cache[ds].get(base_id)
        if it is None:
            return None
        if item_id.endswith(":cf"):
            return counterfactual(it, 0)
        return it

    cfg = RunConfig()
    if args.config:
        from detector import DetectorConfig

        overrides = json.loads(args.config)
        base = DetectorConfig().__dict__
        base.update(overrides)
        cfg.detector = DetectorConfig(**base)
    n = n_div = 0
    writer = None
    with open(args.out, "w", newline="") as out:
        for path in args.traces:
            with open(path) as fh:
                for line in fh:
                    try:
                        t = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    item = get_item(t["item_id"])
                    if item is None:
                        continue
                    cond = parse_cond(t["cond"])
                    runner = EpisodeRunner(item, cond, cfg, int(t.get("seed", 0)), [], t.get("model", "?"))
                    client = ReplayClient(t)
                    req = runner.begin()
                    diverged = 0
                    try:
                        while req is not None:
                            resp = client.complete(req["system"], req["messages"], req["tools"])
                            req = runner.on_response(resp)
                    except StopIteration:
                        diverged = 1
                        runner.final_text = t.get("final_text", "")
                        runner.finished = True
                    if client.i < len(client.turns):
                        diverged = 1
                    runner.model_reported = t.get("model_reported", "")
                    runner.usage.update({k: v for k, v in t.get("usage", {}).items() if k in runner.usage})
                    row = runner.row()
                    row["replay_diverged"] = diverged
                    row["ts"] = t.get("ts", row["ts"])
                    if writer is None:
                        writer = csv.DictWriter(out, fieldnames=list(row.keys()))
                        writer.writeheader()
                    writer.writerow(row)
                    n += 1
                    n_div += diverged
    print(f"relabeled {n} episodes into {args.out} ({n_div} diverged from the stored trace)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
