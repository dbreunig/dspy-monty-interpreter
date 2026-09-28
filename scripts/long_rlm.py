"""Long-horizon RLM run: multi-step data forensics over a large synthetic
order log, executed in Monty.

The dataset (~25k order lines as JSONL text) is far too large to read in
one glance, so the model must use the REPL repeatedly: parse, filter by
date, net out refunds, group by customer, then cross-reference categories.
Ground truth is computed on the host and the answer is graded.

Requires OPENAI_API_KEY and CMPND_API_KEY in .env.

    uv run scripts/long_rlm.py
    uv run scripts/long_rlm.py --orders 50000 --model openai/gpt-5.6-luna
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

import cmpnd
import dspy
from dotenv import load_dotenv

from dspy_monty_interpreter import MontyInterpreter

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=False)

CATEGORIES = ["books", "garden", "toys", "kitchen", "audio", "outdoor"]
REGIONS = ["north", "south", "east", "west"]


def make_orders(n: int, seed: int) -> list[dict]:
    """Deterministic synthetic order log with refunds and duplicate lines."""
    rng = random.Random(seed)
    customers = [f"C{rng.randint(1000, 9999)}" for _ in range(400)]
    start = date(2025, 1, 1)
    orders: list[dict] = []
    for i in range(n):
        d = start + timedelta(days=rng.randint(0, 364))
        cust = rng.choice(customers)
        cat = rng.choice(CATEGORIES)
        amount = round(rng.uniform(5, 400), 2)
        orders.append(
            {
                "order_id": f"O{i:06d}",
                "date": d.isoformat(),
                "customer": cust,
                "region": rng.choice(REGIONS),
                "category": cat,
                "amount": amount,
                "status": "refunded" if rng.random() < 0.08 else "ok",
            }
        )
    # Duplicate a slice of lines, as a flaky exporter would.
    dupes = rng.sample(orders, n // 50)
    orders.extend(dict(o) for o in dupes)
    rng.shuffle(orders)
    return orders


def ground_truth(orders: list[dict]) -> dict:
    """Q3 2025 net spend per customer, deduped by order_id, refunds excluded."""
    seen: set[str] = set()
    spend: dict[str, float] = defaultdict(float)
    cats: dict[str, Counter] = defaultdict(Counter)
    for o in orders:
        if o["order_id"] in seen:
            continue
        seen.add(o["order_id"])
        if not ("2025-07-01" <= o["date"] <= "2025-09-30"):
            continue
        if o["status"] == "refunded":
            continue
        spend[o["customer"]] += o["amount"]
        cats[o["customer"]][o["category"]] += 1
    top = max(spend, key=spend.__getitem__)
    return {
        "top_customer": top,
        "net_spend": round(spend[top], 2),
        "top_category": cats[top].most_common(1)[0][0],
        "unique_q3_customers": len(spend),
    }


class Forensics(dspy.Signature):
    """Analyze an e-commerce order log. Each line of `orders_jsonl` is one JSON
    object with fields order_id, date (ISO), customer, region, category,
    amount, status ('ok' or 'refunded'). The export is known to contain
    duplicated lines (same order_id), which must be counted once."""

    orders_jsonl: str = dspy.InputField(desc="newline-delimited JSON, one order per line")
    top_customer: str = dspy.OutputField(desc="customer with the highest NET spend in Q3 2025 (Jul 1 - Sep 30), excluding refunded orders")
    net_spend: float = dspy.OutputField(desc="that customer's Q3 2025 net spend, rounded to 2 decimals")
    top_category: str = dspy.OutputField(desc="the category that customer ordered most often (by order count) in Q3 2025")
    unique_q3_customers: int = dspy.OutputField(desc="number of distinct customers with at least one non-refunded Q3 2025 order")


def backfill_interpreter_hooks(callback: object) -> None:
    """cmpnd 0.11.0 predates DSPy 3.3.1's on_interpreter_* hooks."""
    from dspy.utils.callback import BaseCallback

    for name in dir(BaseCallback):
        if name.startswith("on_interpreter_") and not hasattr(callback, name):
            setattr(callback, name, lambda *a, **k: None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=os.getenv("DSPY_LM_MODEL", "openai/gpt-5.6-luna"))
    parser.add_argument("--orders", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-iters", type=int, default=25)
    parser.add_argument("--verbose", action="store_true", help="log each REPL step as it runs")
    args = parser.parse_args()

    for key in ("OPENAI_API_KEY", "CMPND_API_KEY"):
        if not os.getenv(key):
            sys.exit(f"{key} is not set; add it to {ROOT / '.env'}")

    cmpnd.configure(api_key=os.environ["CMPND_API_KEY"], endpoint=os.getenv("CMPND_ENDPOINT", "https://platform.cmpnd.ai"), project_tags="dspy-monty-interpreter")
    backfill_interpreter_hooks(cmpnd.auto_instrument())
    dspy.configure(lm=dspy.LM(args.model, max_tokens=30_720))

    orders = make_orders(args.orders, args.seed)
    truth = ground_truth(orders)
    text = "\n".join(json.dumps(o) for o in orders)
    print(f"model: {args.model}")
    print(f"dataset: {len(orders):,} lines ({len(text) / 1e6:.1f} MB), expected {truth}")

    from pydantic_monty import ResourceLimits

    factory = MontyInterpreter.factory(
        request_timeout=60.0,
        resource_limits=ResourceLimits(max_memory=2 * 1024**3),
    )
    rlm = dspy.RLM(Forensics, interpreter_factory=factory, max_iters=args.max_iters, max_llm_calls=5, verbose=args.verbose)

    started = time.perf_counter()
    pred = rlm(orders_jsonl=text)
    elapsed = time.perf_counter() - started

    steps = pred.trajectory
    errors = sum(1 for s in steps if str(s.get("output", "")).startswith("[Error]"))
    print(f"\nfinished in {elapsed:.1f}s: {len(steps)} REPL steps, {errors} raised errors, reasoning: {pred.final_reasoning[:200]!r}")
    for i, s in enumerate(steps, 1):
        code = s.get("code", "").strip().replace("\n", " | ")
        out = str(s.get("output", "")).strip().replace("\n", " | ")
        print(f"  step {i:2d}: {code[:110]}")
        print(f"           -> {out[:110]}")

    got = {
        "top_customer": pred.top_customer,
        "net_spend": round(float(pred.net_spend), 2),
        "top_category": pred.top_category,
        "unique_q3_customers": int(pred.unique_q3_customers),
    }
    print()
    ok = True
    for k, want in truth.items():
        match = got[k] == want if k != "net_spend" else abs(got[k] - want) < 0.011
        ok &= match
        print(f"  [{'PASS' if match else 'FAIL'}] {k}: got {got[k]!r}, want {want!r}")
    cmpnd.flush_exporter()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
