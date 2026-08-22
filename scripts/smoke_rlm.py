"""Live smoke test: dspy.RLM + MontyInterpreter + cmpnd tracing.

Runs three RLM programs against a real LM, each exercising a different
part of the adapter, and ships traces (LM calls, interpreter lifecycle,
sandbox tool calls) to cmpnd.

Requires OPENAI_API_KEY and CMPND_API_KEY in .env (or the environment).

    uv run scripts/smoke_rlm.py
    uv run scripts/smoke_rlm.py --model openai/gpt-5.6-luna
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cmpnd
import dspy
from dotenv import load_dotenv

from dspy_monty_interpreter import MontyInterpreter

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=False)


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        sys.exit(f"{name} is not set; add it to {ROOT / '.env'}")
    return value


def lookup_city_population(city: str) -> str:
    """Return the population of a city from a tiny stub database."""
    return {"tokyo": "13960000", "paris": "2161000", "lagos": "15388000"}.get(city.lower(), "0")


def _backfill_interpreter_hooks(callback: object) -> None:
    """cmpnd 0.11.0's CmpndCallback does not subclass dspy's BaseCallback, so it
    lacks the no-op defaults for the interpreter lifecycle hooks DSPy 3.3.1
    added. DSPy logs a warning per missing hook; give it no-ops until cmpnd
    ships its own handlers (at which point this is a no-op itself)."""
    from dspy.utils.callback import BaseCallback

    for name in dir(BaseCallback):
        if name.startswith("on_interpreter_") and not hasattr(callback, name):
            setattr(callback, name, lambda *args, **kwargs: None)


def check(label: str, ok: bool, detail: str) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
    return ok


def run_product(model: str) -> bool:
    """Factory-created interpreter; pure computation in Monty."""
    print("\n1. product via interpreter_factory")
    rlm = dspy.RLM(
        "numbers: list[int] -> product: int",
        interpreter_factory=MontyInterpreter,
        max_iters=5,
        max_llm_calls=3,
    )
    result = rlm(numbers=[2, 3, 5, 7])
    return check("product", int(result.product) == 210, f"got {result.product!r}, want 210")


def run_tool(model: str) -> bool:
    """User tool invoked from the sandbox (fires on_interpreter_tool_call_*)."""
    print("\n2. tool call through Monty external_lookup")
    rlm = dspy.RLM(
        "city: str -> population: int",
        interpreter_factory=MontyInterpreter,
        tools=[lookup_city_population],
        max_iters=5,
        max_llm_calls=3,
    )
    result = rlm(city="Paris")
    return check("population", int(result.population) == 2161000, f"got {result.population!r}, want 2161000")


def run_error_recovery(model: str) -> bool:
    """Caller-owned interpreter reused across calls; a task that tends to
    provoke a runtime error first (unknown key), so RLM must recover from a
    CodeExecutionError correction turn rather than aborting."""
    print("\n3. caller-owned interpreter + error recovery")
    interpreter = MontyInterpreter(request_timeout=30.0)
    rlm = dspy.RLM(
        "records: dict, key: str -> value: str",
        max_iters=6,
        max_llm_calls=3,
    )
    try:
        records = {"alpha": "1", "beta": "2"}
        r1 = rlm(interpreter, records=records, key="beta")
        ok = check("value(beta)", r1.value == "2", f"got {r1.value!r}, want '2'")
        r2 = rlm(interpreter, records=records, key="gamma")
        # Any non-crashing answer is fine; the point is forward() completed
        # after the sandbox raised a KeyError or similar along the way.
        ok &= check("value(gamma) completed", isinstance(r2.value, str), f"got {r2.value!r}")
        return ok
    finally:
        interpreter.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=os.getenv("DSPY_LM_MODEL", "openai/gpt-5.6-luna"))
    args = parser.parse_args()

    require_env("OPENAI_API_KEY")
    cmpnd.configure(
        api_key=require_env("CMPND_API_KEY"),
        endpoint=os.getenv("CMPND_ENDPOINT", "https://platform.cmpnd.ai"),
        project_tags="dspy-monty-interpreter",
    )
    callback = cmpnd.auto_instrument()  # registers a DSPy callback
    _backfill_interpreter_hooks(callback)

    dspy.configure(lm=dspy.LM(args.model, max_tokens=30_720))
    print(f"model: {args.model}")
    print(f"execution_instructions: {MontyInterpreter.execution_instructions[:80]}...")

    results = []
    started = time.perf_counter()
    for step in (run_product, run_tool, run_error_recovery):
        try:
            results.append(step(args.model))
        except Exception as e:  # noqa: BLE001 - report and keep going
            results.append(check(step.__name__, False, f"raised {type(e).__name__}: {e}"))
    elapsed = time.perf_counter() - started

    cmpnd.flush_exporter()
    passed = sum(results)
    print(f"\n{passed}/{len(results)} passed in {elapsed:.1f}s; traces sent to cmpnd")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
