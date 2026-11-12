"""GEPA optimizing a dspy.Flex module that runs in Monty.

The task is synthetic order totals: each order describes line items,
quantities (some spelled out, some in dozens), a coupon, and a sales tax
rate in prose. Bulk line discounts, the coupon, tax, and a shipping
threshold combine into the amount due in cents. One LM call often gets the
arithmetic wrong, so GEPA has room to rewrite the Flex module into steps:
extract the numbers with a predictor and compute the total in Python.

The student model runs the module; the teacher model reflects and proposes
new module source. Both are served by LunaRoute. The script reports the
baseline and optimized validation scores, the optimized module source, and
how each candidate GEPA proposed fared when run in Monty, which shows how
often proposals use features Monty lacks.

Reads LUNAROUTE_API_KEY from ../grand-comfort-bench/.env (or --env).

    uv run --extra dev scripts/flex_gepa_poc.py
    uv run --extra dev scripts/flex_gepa_poc.py --max-metric-calls 600 --train 50 --val 30
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import dspy
from dotenv import load_dotenv

from dspy_monty_interpreter import MontyInterpreter

ROOT = Path(__file__).resolve().parent.parent
LUNAROUTE_BASE = "https://gw.lunaroute.com/v1"

CATALOG = [
    ("box of ballpoint pens", "boxes of ballpoint pens", 425),
    ("ream of copy paper", "reams of copy paper", 899),
    ("desk lamp", "desk lamps", 3450),
    ("stapler", "staplers", 1275),
    ("pack of sticky notes", "packs of sticky notes", 349),
    ("whiteboard marker set", "whiteboard marker sets", 1099),
    ("ergonomic mouse", "ergonomic mice", 2999),
    ("monitor stand", "monitor stands", 4500),
    ("file cabinet", "file cabinets", 12900),
    ("pack of AA batteries", "packs of AA batteries", 799),
]
NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]
DOZENS = {6: "half a dozen", 12: "a dozen", 24: "two dozen"}
TAX_RATES = ["0", "6.25", "7.5", "8.875"]
BULK_UNITS, BULK_PCT = 10, 10
SHIPPING, FREE_SHIPPING_AT = 795, 7500


class OrderTotal(dspy.Signature):
    """Compute the amount the customer pays for an office-supply order, in cents.

    Rules, in order:
    1. A line with 10 or more units gets 10% off that line, rounded to the nearest cent (halves up).
    2. The merchandise subtotal is the sum of the line totals.
    3. A percentage coupon takes that percent off the merchandise subtotal, rounded to the nearest cent
       (halves up). A fixed coupon applies only when the merchandise subtotal exceeds its threshold.
    4. Sales tax is charged on the discounted subtotal, rounded to the nearest cent (halves up).
    5. Shipping is $7.95, free when the discounted subtotal is $75.00 or more. Shipping is not taxed.
    The amount due is the discounted subtotal plus tax plus shipping."""

    order: str = dspy.InputField(desc="The order as the customer wrote it.")
    total_cents: int = dspy.OutputField(desc="Amount due in cents.")


def dollars(cents: int) -> str:
    return f"${Decimal(cents) / 100:,.2f}"


def percent_of(cents: int, pct: Decimal | int | str) -> int:
    return int((Decimal(cents) * Decimal(pct) / 100).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def quantity_text(qty: int, rng: random.Random) -> str:
    if qty in DOZENS and rng.random() < 0.7:
        return DOZENS[qty]
    if qty < len(NUMBER_WORDS) and rng.random() < 0.5:
        return NUMBER_WORDS[qty]
    return str(qty)


def make_example(rng: random.Random) -> dspy.Example:
    """One order in prose, with its total and a breakdown for feedback."""
    line_totals, phrases = [], []
    for singular, plural, price in rng.sample(CATALOG, rng.randint(3, 6)):
        qty = rng.choice([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 20, 24, 30])
        gross = qty * price
        line_totals.append(gross - (percent_of(gross, BULK_PCT) if qty >= BULK_UNITS else 0))
        name = singular if qty == 1 else plural
        phrases.append(f"{quantity_text(qty, rng)} {name} at {dollars(price)} each")
    subtotal = sum(line_totals)

    kind = rng.choice(["none", "percent", "fixed"])
    if kind == "percent":
        pct = rng.choice([5, 10, 15, 20])
        discount = percent_of(subtotal, pct)
        discount_text = f"I have a coupon for {pct}% off."
    elif kind == "fixed":
        off, threshold = rng.choice([(500, 5000), (1000, 10000), (2500, 20000)])
        discount = off if subtotal > threshold else 0
        discount_text = f"My coupon takes {dollars(off)} off orders over {dollars(threshold)}."
    else:
        discount, discount_text = 0, ""
    rate = rng.choice(TAX_RATES)
    taxable = subtotal - discount
    tax = percent_of(taxable, rate)
    shipping = 0 if taxable >= FREE_SHIPPING_AT else SHIPPING
    total = taxable + tax + shipping

    order = f"I'd like {', '.join(phrases[:-1])} and {phrases[-1]}. {discount_text} Sales tax here is {rate}%."
    breakdown = (
        f"line totals {line_totals} cents, merchandise subtotal {subtotal}, discount {discount}, "
        f"discounted subtotal {taxable}, tax {tax} at {rate}%, shipping {shipping}, total {total}"
    )
    return dspy.Example(order=" ".join(order.split()), total_cents=total, breakdown=breakdown).with_inputs("order")


def make_split(n: int, seed: int) -> list[dspy.Example]:
    rng = random.Random(seed)
    return [make_example(rng) for _ in range(n)]


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Exact match on the total, with the gold breakdown as feedback."""
    got = getattr(pred, "total_cents", None)
    try:
        got = int(got)
    except (TypeError, ValueError):
        return dspy.Prediction(score=0.0, feedback=f"No integer total_cents was returned (got {got!r}).")
    if got == gold.total_cents:
        return dspy.Prediction(score=1.0, feedback="Correct.")
    return dspy.Prediction(
        score=0.0,
        feedback=f"Wrong total: returned {got} cents, expected {gold.total_cents}. Correct breakdown: {gold.breakdown}.",
    )


def score(gold, pred, trace=None) -> float:
    return float(metric(gold, pred).score)


class Program(dspy.Module):
    def __init__(self):
        super().__init__()
        self.solve = dspy.Flex(OrderTotal)

    def forward(self, order: str):
        return self.solve(order=order)


def evaluate(program: dspy.Module, devset: list[dspy.Example], threads: int) -> float:
    result = dspy.Evaluate(devset=devset, metric=score, num_threads=threads, display_progress=False)(program)
    return float(result.score)


def probe(program: dspy.Module, examples: list[dspy.Example]) -> dict:
    """Run a candidate on a few examples and record how it fared in Monty."""
    outcomes = []
    for ex in examples:
        try:
            pred = program(order=ex.order)
            outcomes.append("correct" if score(ex, pred) == 1.0 else "wrong")
        except Exception as e:
            first = (str(e).strip().splitlines() or [""])[0][:200]
            outcomes.append(f"error: {type(e).__name__}: {first}")
    return {"outcomes": outcomes, "module_src": program.solve.module_src}


def lunaroute_lm(model: str, key: str) -> dspy.LM:
    return dspy.LM(f"openai/{model}", api_base=LUNAROUTE_BASE, api_key=key)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", type=Path, default=ROOT.parent / "grand-comfort-bench" / ".env")
    parser.add_argument("--student", default="deepseek-4.1-flash")
    parser.add_argument("--teacher", default="glm-5.3")
    parser.add_argument("--train", type=int, default=30)
    parser.add_argument("--val", type=int, default=20)
    parser.add_argument("--max-metric-calls", type=int, default=300)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=ROOT / "tmp" / "flex_gepa_poc")
    args = parser.parse_args()

    load_dotenv(args.env, override=False)
    key = os.getenv("LUNAROUTE_API_KEY")
    if not key:
        print(f"LUNAROUTE_API_KEY is not set (looked in {args.env}).", file=sys.stderr)
        return 1

    student, teacher = lunaroute_lm(args.student, key), lunaroute_lm(args.teacher, key)
    dspy.configure(lm=student, interpreter_factory=MontyInterpreter.factory(request_timeout=60.0))

    trainset = make_split(args.train, args.seed)
    valset = make_split(args.val, args.seed + 1)
    run_dir = args.out / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"student={args.student} teacher={args.teacher} train={len(trainset)} val={len(valset)}")
    baseline = Program()
    started = time.time()
    baseline_score = evaluate(baseline, valset, args.threads)
    print(f"baseline val score: {baseline_score:.1f}%")

    gepa = dspy.GEPA(
        metric=metric,
        reflection_lm=teacher,
        max_metric_calls=args.max_metric_calls,
        num_threads=args.threads,
        track_stats=True,
        log_dir=str(run_dir / "gepa"),
        seed=args.seed,
    )
    optimized = gepa.compile(baseline, trainset=trainset, valset=valset)
    optimized_score = evaluate(optimized, valset, args.threads)
    print(f"optimized val score: {optimized_score:.1f}%")
    print("\noptimized module_src:\n" + (optimized.solve.module_src or ""))

    candidates = []
    details = getattr(optimized, "detailed_results", None)
    for i, candidate in enumerate(getattr(details, "candidates", []) or []):
        try:
            report = probe(candidate, valset[:3])
        except Exception:
            report = {"outcomes": [f"probe failed: {traceback.format_exc(limit=1)}"], "module_src": None}
        report["val_score"] = details.val_aggregate_scores[i]
        candidates.append(report)
        print(f"candidate {i}: val {report['val_score']:.2f} probe {report['outcomes']}")

    result = {
        "student": args.student,
        "teacher": args.teacher,
        "train": len(trainset),
        "val": len(valset),
        "max_metric_calls": args.max_metric_calls,
        "baseline_val": baseline_score,
        "optimized_val": optimized_score,
        "optimized_module_src": optimized.solve.module_src,
        "candidates": candidates,
        "seconds": round(time.time() - started, 1),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    (run_dir / "optimized_module.py").write_text(optimized.solve.module_src or "")
    print(f"\nwrote {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
