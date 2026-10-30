#!/usr/bin/env python3
"""Batches wave mode: every open episode's next model call goes into one Message Batch (50 percent of list price);
the results are applied, tool calls are executed locally, and the next wave is submitted, until every episode is done.

  # dry run of the wave machinery with the scripted mock (no API):
  python3 -I run_batches.py --backend mock --mini --cells REQUIRED:NEUTRAL --out waves.csv

  # real run (reads the key from P2_KEY_FILE, default ~/.config/untraceable-numbers/anthropic_key; never printed):
  python3 run_batches.py --backend anthropic --model claude-haiku-5-5 --datasets finqa,tatqa --n-per-bucket 5 \
      --cells REQUIRED:NEUTRAL,REQUIRED:BREVITY,REQUIRED:FAILURE --out pilot.csv --traces pilot.jsonl \
      --cost-log ../results/cost-log.csv --cap-usd 80 --tag pilot-haiku

Spending guard: before each wave the projected cost (request size x list price x batch factor, plus an assumed output per
turn) is added to the cumulative spend in --cost-log; the wave is refused when the total would pass --cap-usd. After each
wave the actual usage is priced and appended to the cost log. State is pickled after each wave (--state), so a crashed or
interrupted run resumes with --resume. The key file is read inside the process and passed to the SDK client; it is never
printed, logged or written anywhere.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import PRESSURES, REGIMES, Condition, EpisodeRunner, RunConfig  # noqa: E402
from clients import LLMResponse, MockClient, MockPolicy, _strip_private_keys  # noqa: E402
from loaders import bucket_counts, counterfactual, load_dataset, load_mini_sample, stratified_sample  # noqa: E402

HERE = Path(__file__).resolve().parent
PRICES = {  # USD per million tokens, list, cached 2026-10-06 from the API reference
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}
BATCH_FACTOR = 0.5
CACHE_READ_FACTOR = 0.1
CACHE_WRITE_FACTOR = 1.25


# ----------------------------------------------------------------------------- cost ledger

def price_usage(model: str, usage: dict[str, int], batch: bool = True) -> float:
    pi, po = PRICES.get(model, (0.0, 0.0))
    f = BATCH_FACTOR if batch else 1.0
    return f * (usage.get("input_tokens", 0) * pi + usage.get("output_tokens", 0) * po
                + usage.get("cache_read_input_tokens", 0) * pi * CACHE_READ_FACTOR
                + usage.get("cache_creation_input_tokens", 0) * pi * CACHE_WRITE_FACTOR) / 1e6


def ledger_total(path: Path) -> float:
    if not path.exists():
        return 0.0
    tot = 0.0
    with path.open() as fh:
        for r in csv.DictReader(fh):
            try:
                tot += float(r["cost_usd"])
            except (KeyError, ValueError):
                pass
    return tot


def ledger_append(path: Path, row: dict[str, Any]) -> None:
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["ts", "tag", "batch_id", "model", "requests", "succeeded", "errored", "input_tokens", "output_tokens",
                                           "cache_read_input_tokens", "cache_creation_input_tokens", "cost_usd", "cumulative_usd", "note"])
        if new:
            w.writeheader()
        w.writerow(row)


def project_wave_cost(model: str, requests: list[dict[str, Any]], out_per_turn: float) -> float:
    pi, po = PRICES.get(model, (0.0, 0.0))
    tokens_in = sum(max(1, len(json.dumps(r["params"], default=str)) // 4) for r in requests)
    return BATCH_FACTOR * (tokens_in * pi + len(requests) * out_per_turn * po) / 1e6


# ----------------------------------------------------------------------------- backends

class MockBatchBackend:
    """Runs each request through a per-episode MockClient at once; same interface as the real backend."""

    def __init__(self, policy: MockPolicy) -> None:
        self.policy = policy
        self.clients: dict[str, MockClient] = {}
        self.model_name = "mock-model"

    def begin_episode(self, key: str, item: Any, cond: Condition, seed: int) -> None:
        c = MockClient(policy=self.policy)
        c.begin_episode(item, cond, seed)
        self.clients[key] = c

    def truth(self, key: str) -> dict[str, Any]:
        return self.clients[key].truth

    def submit_and_collect(self, requests: list[dict[str, Any]], poll_s: int = 0) -> tuple[str, dict[str, Any]]:
        out: dict[str, Any] = {}
        for r in requests:
            c = self.clients[r["episode"]]
            p = r["params"]
            resp = c.complete(p["system"], p["messages"], p.get("tools", []), max_tokens=p["max_tokens"], effort=p["effort"],
                              temperature=p.get("temperature"), json_schema=p.get("json_schema"))
            out[r["custom_id"]] = {"ok": True, "resp": resp}
        return "mock-batch", out

    def model_info(self) -> dict[str, Any]:
        return {"id": "mock-model"}


class AnthropicBatchBackend:
    def __init__(self, model: str, key_file: str, workspace_file: str | None = None) -> None:
        import anthropic  # the SDK is installed in the run environment, not in the harness

        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        self._anthropic = anthropic
        self._Params = MessageCreateParamsNonStreaming
        self._Request = Request
        key = Path(os.path.expanduser(key_file)).read_text().strip()
        if not key:
            raise SystemExit("empty key file")
        headers: dict[str, str] = {}
        if workspace_file:
            wpath = Path(os.path.expanduser(workspace_file))
            if wpath.exists():
                wid = wpath.read_text().strip()
                if wid:
                    headers["anthropic-workspace-id"] = wid  # the key is not scoped to a workspace; the header selects it
        self.client = anthropic.Anthropic(api_key=key, default_headers=headers or None)
        del key
        self.model_name = model

    def begin_episode(self, key: str, item: Any, cond: Condition, seed: int) -> None:
        return None

    def truth(self, key: str) -> dict[str, Any]:
        return {}

    def model_info(self) -> dict[str, Any]:
        try:
            m = self.client.models.retrieve(self.model_name)
            return {"id": m.id, "display_name": getattr(m, "display_name", ""), "created_at": str(getattr(m, "created_at", ""))}
        except Exception as exc:  # noqa: BLE001
            return {"id": self.model_name, "error": str(exc)[:100]}

    def _params(self, p: dict[str, Any]) -> Any:
        kwargs: dict[str, Any] = {"model": self.model_name, "max_tokens": p["max_tokens"], "system": p["system"],
                                  "messages": _strip_private_keys(p["messages"]), "output_config": {"effort": p["effort"]}}
        if p.get("tools"):
            kwargs["tools"] = p["tools"]
        if p.get("json_schema") is not None:
            kwargs["output_config"]["format"] = {"type": "json_schema", "schema": p["json_schema"]}
        if p.get("temperature") is not None:
            kwargs["temperature"] = p["temperature"]
        return self._Params(**kwargs)

    def submit_and_collect(self, requests: list[dict[str, Any]], poll_s: int = 30) -> tuple[str, dict[str, Any]]:
        batch = self.client.messages.batches.create(requests=[self._Request(custom_id=r["custom_id"], params=self._params(r["params"])) for r in requests])
        bid = batch.id
        print(f"  batch {bid} submitted with {len(requests)} requests; polling every {poll_s}s", flush=True)
        t0 = time.time()
        while True:
            b = self.client.messages.batches.retrieve(bid)
            if b.processing_status == "ended":
                break
            rc = b.request_counts
            print(f"  {int(time.time() - t0)}s: {b.processing_status}, processing {rc.processing}, succeeded {rc.succeeded}, errored {rc.errored}", flush=True)
            time.sleep(poll_s)
        out: dict[str, Any] = {}
        for result in self.client.messages.batches.results(bid):
            rt = result.result.type
            if rt == "succeeded":
                msg = result.result.message
                blocks = [b.model_dump(exclude_none=True) for b in msg.content]  # passed back unchanged on the next turn (incl. thinking signatures)
                usage = {"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens,
                         "cache_read_input_tokens": getattr(msg.usage, "cache_read_input_tokens", 0) or 0,
                         "cache_creation_input_tokens": getattr(msg.usage, "cache_creation_input_tokens", 0) or 0}
                out[result.custom_id] = {"ok": True, "resp": LLMResponse(blocks=blocks, stop_reason=msg.stop_reason, usage=usage, model=msg.model)}
            else:
                err = ""
                if rt == "errored":
                    e = result.result.error
                    err = f"{getattr(e, 'type', '')}: {str(getattr(e, 'message', e))[:160]}"
                out[result.custom_id] = {"ok": False, "type": rt, "error": err or rt}
        return bid, out


# ----------------------------------------------------------------------------- wave loop

def parse_cells(spec: str) -> list[tuple[str, str]]:
    if spec in ("all", ""):
        return [(r, p) for r in REGIMES for p in PRESSURES]
    cells = []
    for part in spec.split(","):
        r, p = part.strip().split(":")
        if r not in REGIMES or p not in PRESSURES:
            raise SystemExit(f"unknown cell {part!r}")
        cells.append((r, p))
    return cells


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="mock", choices=["mock", "anthropic"])
    ap.add_argument("--model", default=os.environ.get("P2_MODEL", "claude-haiku-5-5"))
    ap.add_argument("--key-file", default=os.environ.get("P2_KEY_FILE", "~/.config/untraceable-numbers/anthropic_key"))
    ap.add_argument("--workspace-file", default=os.environ.get("P2_WORKSPACE_FILE", "~/.config/untraceable-numbers/anthropic_workspace_id"))
    ap.add_argument("--datasets", default="finqa,tatqa")
    ap.add_argument("--mini", action="store_true")
    ap.add_argument("--n-per-bucket", type=int, default=5)
    ap.add_argument("--buckets", default="1,2,3+")
    ap.add_argument("--cells", default="REQUIRED:NEUTRAL")
    ap.add_argument("--efforts", default="low")
    ap.add_argument("--fmts", default="FREE")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--counterfactual", action="store_true")
    ap.add_argument("--out", default="waves.csv")
    ap.add_argument("--traces", default=None)
    ap.add_argument("--state", default=None, help="pickle path for resumable state (default <out>.state.pkl)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--cost-log", default=str(HERE.parent / "cost-log.csv"))
    ap.add_argument("--cap-usd", type=float, default=float(os.environ.get("P2_HARD_CAP_USD", "80")))
    ap.add_argument("--out-per-turn", type=float, default=600.0, help="assumed output tokens per turn for the projection (incl. thinking)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--poll-s", type=int, default=30)
    ap.add_argument("--max-batch", type=int, default=5000)
    ap.add_argument("--max-waves", type=int, default=12)
    ap.add_argument("--mock-leak-p", type=float, default=None)
    args = ap.parse_args()

    state_path = Path(args.state or (args.out + ".state.pkl"))
    out = Path(args.out)
    cost_log = Path(args.cost_log)
    cfg = RunConfig()

    if args.backend == "mock":
        policy = MockPolicy(seed=args.seed)
        if args.mock_leak_p is not None:
            policy.leak_p = args.mock_leak_p
        backend: Any = MockBatchBackend(policy)
        batch_pricing_model = "claude-haiku-5-5"  # price the mock as the small tier so the ledger logic is exercised
    else:
        backend = AnthropicBatchBackend(args.model, args.key_file, args.workspace_file)
        batch_pricing_model = args.model
    model_name = backend.model_name if args.backend == "mock" else args.model

    if args.resume and state_path.exists():
        with state_path.open("rb") as fh:
            st = pickle.load(fh)
        runners, pending, wave, done_keys = st["runners"], st["pending"], st["wave"], st["done_keys"]
        print(f"resumed: {len(pending)} open episodes at wave {wave}, {len(done_keys)} done")
    else:
        buckets = tuple(b.strip() for b in args.buckets.split(","))
        items: list[Any] = []
        pools: dict[str, list[Any]] = {}
        if args.mini:
            pool = load_mini_sample(HERE / "sample_data" / "mini_finqa.json")
            items.extend(stratified_sample(pool, args.n_per_bucket, args.seed, buckets))
            pools["finqa"] = pool
        else:
            for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
                pool = load_dataset(ds)
                chosen = stratified_sample(pool, args.n_per_bucket, args.seed, buckets)
                print(f"{ds}: loaded {len(pool)}, buckets {bucket_counts(pool)}; sampled {len(chosen)}")
                items.extend(chosen)
                pools[ds] = pool
        if args.counterfactual:
            cf = [c for c in (counterfactual(it, args.seed) for it in items if not it.history) if c is not None]
            print(f"counterfactual: {len(cf)} of {len(items)} items perturbed")
            items = cf
        cells = parse_cells(args.cells)
        conds = [Condition(r, p, e, f) for (r, p) in cells for e in args.efforts.split(",") for f in args.fmts.split(",")]
        done_keys: set[str] = set()
        if out.exists():
            with out.open() as fh:
                for r in csv.DictReader(fh):
                    done_keys.add(f"{r['item_id']}|{r['regime']}|{r['pressure']}|{r['effort']}|{r['fmt']}|{r['model']}|{r['seed']}")
        rng = random.Random(args.seed)
        runners: dict[str, EpisodeRunner] = {}
        pending: dict[str, dict[str, Any]] = {}
        for cond in conds:
            for it in items:
                for rep in range(args.repeats):
                    seed = args.seed + rep
                    key = f"{it.id}|{cond.key()}|{model_name}|{seed}"
                    if key in done_keys:
                        continue
                    pool = [d for d in pools.get(it.dataset, []) if d.id != it.id and d.context_text != it.context_text]
                    distractors = rng.sample(pool, min(cfg.distract_tables, len(pool))) if cond.pressure == "DISTRACT" else []
                    runner = EpisodeRunner(it, cond, cfg, seed, distractors, model_name)
                    backend.begin_episode(key, it, cond, seed)
                    runners[key] = runner
                    pending[key] = runner.begin()
        wave = 0
        print(f"{len(pending)} episodes to run over {len(conds)} conditions ({len(done_keys)} already in {out})")

    info = backend.model_info()
    print(f"model: {info}")
    writer = None
    fh = out.open("a", newline="")
    trace_fh = open(args.traces, "a") if args.traces else None
    retries: dict[str, int] = {}

    def finish(key: str) -> None:
        nonlocal writer
        runner = runners[key]
        runner.truth = backend.truth(key)
        row = runner.row()
        if writer is None:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if out.stat().st_size == 0:
                writer.writeheader()
        writer.writerow(row)
        fh.flush()
        if trace_fh is not None:
            trace_fh.write(json.dumps(runner.trace(), default=str) + "\n")
            trace_fh.flush()
        pending.pop(key, None)
        done_keys.add(key)

    try:
        while pending and wave < args.max_waves:
            wave += 1
            keys = sorted(pending)
            id_map: dict[str, str] = {}
            requests = []
            for k in keys:
                cid = "e" + hashlib.sha1(f"{k}|w{wave}".encode()).hexdigest()[:20]
                id_map[cid] = k
                requests.append({"custom_id": cid, "episode": k, "params": pending[k]})
            spent = ledger_total(cost_log)
            projected = project_wave_cost(batch_pricing_model, requests, args.out_per_turn)
            print(f"wave {wave}: {len(requests)} requests; spent so far {spent:.2f} USD; projected for this wave {projected:.2f} USD (cap {args.cap_usd:.2f})", flush=True)
            if spent + projected > args.cap_usd:
                print(f"STOP: projected spend {spent + projected:.2f} USD would pass the cap of {args.cap_usd:.2f} USD; nothing submitted this wave", flush=True)
                break
            for start in range(0, len(requests), args.max_batch):
                chunk = requests[start:start + args.max_batch]
                bid, results = backend.submit_and_collect(chunk, poll_s=args.poll_s)
                usage_tot = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
                ok = err = 0
                for cid, res in results.items():
                    k = id_map.get(cid)
                    if k is None or k not in runners:
                        continue
                    if res["ok"]:
                        ok += 1
                        for u in usage_tot:
                            usage_tot[u] += res["resp"].usage.get(u, 0) or 0
                        nxt = runners[k].on_response(res["resp"])
                        if nxt is None:
                            finish(k)
                        else:
                            pending[k] = nxt
                    else:
                        err += 1
                        if res.get("type") in ("errored", "expired") and "invalid_request" not in res.get("error", "") and retries.get(k, 0) < 2:
                            retries[k] = retries.get(k, 0) + 1  # server-side error: resend the same request next wave
                        else:
                            runners[k].fail(res.get("error", res.get("type", "error")))
                            finish(k)
                cost = price_usage(batch_pricing_model, usage_tot, batch=True) if args.backend != "mock" else price_usage(batch_pricing_model, usage_tot, batch=True)
                cumulative = ledger_total(cost_log) + cost
                ledger_append(cost_log, {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": args.tag or args.backend, "batch_id": bid, "model": model_name,
                                         "requests": len(chunk), "succeeded": ok, "errored": err, **usage_tot, "cost_usd": round(cost, 4),
                                         "cumulative_usd": round(cumulative, 4), "note": f"wave {wave}"})
                print(f"  wave {wave} batch done: {ok} ok, {err} errors, tokens in {usage_tot['input_tokens']:,} out {usage_tot['output_tokens']:,}, cost {cost:.4f} USD, cumulative {cumulative:.2f} USD", flush=True)
            with state_path.open("wb") as sfh:
                pickle.dump({"runners": runners, "pending": pending, "wave": wave, "done_keys": done_keys}, sfh)
            print(f"  open episodes after wave {wave}: {len(pending)}", flush=True)
        if pending and wave >= args.max_waves:
            for k in list(pending):
                runners[k].fail("max waves reached")
                finish(k)
    finally:
        fh.close()
        if trace_fh is not None:
            trace_fh.close()
    print(f"done: {len(done_keys)} episodes in {out}; cumulative spend {ledger_total(cost_log):.2f} USD (cap {args.cap_usd:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
