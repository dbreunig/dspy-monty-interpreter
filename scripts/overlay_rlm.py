"""Live test of overlay mounts through dspy.RLM, traced with cmpnd.

Mounts a read-only reports directory at /data and an overlay scratch
directory at /scratch, then asks the model to build a combined index file
under /scratch and report on it. Overlay writes are discarded when each
execute() ends, so the model has to write and read back within one step;
the prompt's mount section tells it so. Afterwards the host scratch
directory must still be empty.

The action prompt, including the generated "Mounted directories" section,
is visible in cmpnd on the LM-call spans; it is also printed here.

Requires OPENAI_API_KEY and CMPND_API_KEY in .env.

    uv run scripts/overlay_rlm.py
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import cmpnd
import dspy
from dotenv import load_dotenv

from dspy_monty_interpreter import MontyInterpreter, MountDir

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=False)

REPORTS = {
    "q1/sales.txt": "Q1 sales: 120 units in north, 95 in south.\n",
    "q2/sales.txt": "Q2 sales: 140 units in north, 110 in south.\n",
    "q3/forecast.txt": "Q3 forecast: revenue up 12% on stronger north demand.\n",
    "q3/budget.txt": "Q3 budget: headcount flat, marketing spend +5%.\n",
    "readme.md": "Quarterly reports. One directory per quarter.\n",
}


class BuildIndex(dspy.Signature):
    """Build an index of every report under /data. Write the index to
    /scratch/index.txt as one line per file, formatted `<path>: <first line>`,
    then read the file back to verify it."""

    request: str = dspy.InputField()
    index_lines: int = dspy.OutputField(desc="number of lines in /scratch/index.txt after writing it")
    forecast_path: str = dspy.OutputField(desc="full /data path of the file that contains the Q3 forecast")


def backfill_interpreter_hooks(callback: object) -> None:
    """cmpnd 0.11.0 predates DSPy 3.3.1's on_interpreter_* hooks."""
    from dspy.utils.callback import BaseCallback

    for name in dir(BaseCallback):
        if name.startswith("on_interpreter_") and not hasattr(callback, name):
            setattr(callback, name, lambda *a, **k: None)


def check(label: str, ok: bool, detail: str) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=os.getenv("DSPY_LM_MODEL", "openai/gpt-5.6-luna"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    for key in ("OPENAI_API_KEY", "CMPND_API_KEY"):
        if not os.getenv(key):
            sys.exit(f"{key} is not set; add it to {ROOT / '.env'}")
    cmpnd.configure(api_key=os.environ["CMPND_API_KEY"], endpoint=os.getenv("CMPND_ENDPOINT", "https://platform.cmpnd.ai"), project_tags="dspy-monty-interpreter")
    backfill_interpreter_hooks(cmpnd.auto_instrument())
    dspy.configure(lm=dspy.LM(args.model, max_tokens=30_720))

    with tempfile.TemporaryDirectory() as tmp:
        reports = Path(tmp) / "reports"
        scratch = Path(tmp) / "scratch"
        scratch.mkdir()
        for rel, text in REPORTS.items():
            path = reports / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

        factory = MontyInterpreter.factory(
            mounts=[
                MountDir(virtual_path="/data", host_path=str(reports), mode="read-only"),
                MountDir(virtual_path="/scratch", host_path=str(scratch), mode="overlay"),
            ],
            request_timeout=30.0,
        )
        print(f"model: {args.model}\n\nexecution_instructions (mount section):")
        print("  Mounted directories" + factory.execution_instructions.split("Mounted directories")[1].replace("\n", "\n  "))

        rlm = dspy.RLM(BuildIndex, interpreter_factory=factory, max_iters=10, max_llm_calls=2, verbose=args.verbose)
        started = time.perf_counter()
        pred = rlm(request="Index the reports and find the Q3 forecast.")
        elapsed = time.perf_counter() - started

        steps = pred.trajectory
        errors = [s for s in steps if str(s.get("output", "")).startswith("[Error]")]
        print(f"\nfinished in {elapsed:.1f}s: {len(steps)} REPL steps, {len(errors)} raised errors")
        for i, s in enumerate(steps, 1):
            print(f"  step {i:2d}: {s['code'].strip().replace(chr(10), ' | ')[:120]}")
            print(f"           -> {str(s.get('output', '')).strip().replace(chr(10), ' | ')[:120]}")

        wrote_scratch = any("/scratch/index.txt" in s["code"] and ("write" in s["code"] or "'w'" in s["code"] or '"w"' in s["code"]) for s in steps)
        host_files = sorted(p.name for p in scratch.iterdir())
        print()
        ok = check("index_lines", int(pred.index_lines) == len(REPORTS), f"got {pred.index_lines!r}, want {len(REPORTS)}")
        ok &= check("forecast_path", pred.forecast_path.strip() == "/data/q3/forecast.txt", f"got {pred.forecast_path!r}")
        ok &= check("model wrote via overlay", wrote_scratch, "a step opened /scratch/index.txt for writing")
        ok &= check("host scratch untouched", host_files == [], f"host scratch contains {host_files}")
        ok &= check("no Monty errors", not errors, f"{len(errors)} error turns")

    cmpnd.flush_exporter()
    print("\ntraces sent to cmpnd (project tag dspy-monty-interpreter)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
