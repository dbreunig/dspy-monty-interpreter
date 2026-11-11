#!/usr/bin/env python3
"""Benchmark: MontyInterpreter vs DSPy PythonInterpreter.

Measures startup time, per-task execution time, tool call overhead,
state accumulation cost, and total wall-clock time across a diverse
set of workloads.

Usage:
    python benchmarks/bench.py
    python benchmarks/bench.py --runs 5        # average over N runs
    python benchmarks/bench.py --json          # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from dspy.primitives import PythonInterpreter
from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput

from dspy_monty_interpreter import MontyInterpreter

# ---------------------------------------------------------------------------
# Test cases — each is (name, code, variables, expected_output_substring)
# ---------------------------------------------------------------------------

TASKS: list[tuple[str, str, dict[str, Any] | None, str | None]] = [
    # --- Arithmetic & expressions ---
    (
        "arithmetic",
        "print((2**10 + 3**7) * 17 - 42)",
        None,
        "54545",
    ),
    # --- String manipulation ---
    (
        "string ops",
        """\
words = "the quick brown fox jumps over the lazy dog".split()
result = " ".join(sorted(words, key=lambda w: w[::-1]))
print(result)
""",
        None,
        None,  # just check it runs
    ),
    # --- List comprehension & filtering ---
    (
        "list comprehension",
        """\
primes = []
for n in range(2, 200):
    if all(n % d != 0 for d in range(2, int(n**0.5) + 1)):
        primes.append(n)
print(len(primes))
""",
        None,
        "46",
    ),
    # --- Nested loops & computation ---
    (
        "nested loops (dot products)",
        """\
n = 30
A = [[i * n + j for j in range(n)] for i in range(n)]
B = [[j * n + i for j in range(n)] for i in range(n)]
results = []
for i in [0, 15, 29]:
    for j in [0, 15, 29]:
        s = 0
        for k in range(n):
            s += A[i][k] * B[k][j]
        results.append(s)
print(results[0], results[4], results[8])
""",
        None,
        None,
    ),
    # --- Recursion ---
    (
        "recursion (fibonacci)",
        """\
def fib(n):
    if n < 2:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
print(fib(50))
""",
        None,
        "12586269025",
    ),
    # --- Dictionary operations ---
    (
        "dict ops",
        """\
freq = {}
text = "to be or not to be that is the question"
for word in text.split():
    freq[word] = freq.get(word, 0) + 1
top = sorted(freq.items(), key=lambda x: x[1], reverse=True)
print(top[0][0], top[0][1])
""",
        None,
        "to 2",
    ),
    # --- Try/except error handling ---
    (
        "error handling",
        """\
results = []
for val in [10, 0, 5, None, 3]:
    try:
        results.append(100 // val)
    except (ZeroDivisionError, TypeError):
        results.append(-1)
print(results)
""",
        None,
        "[10, -1, 20, -1, 33]",
    ),
    # --- Variable injection ---
    (
        "variable injection",
        """\
total = sum(items)
avg = total / len(items)
print(f"{total} {avg:.1f}")
""",
        {"items": [10, 20, 30, 40, 50]},
        "150 30.0",
    ),
    # --- Higher-order functions ---
    (
        "higher-order functions",
        """\
def compose(f, g):
    def h(x):
        return f(g(x))
    return h

double = lambda x: x * 2
inc = lambda x: x + 1
f = compose(double, inc)
print([f(i) for i in range(5)])
""",
        None,
        "[2, 4, 6, 8, 10]",
    ),
    # --- Data processing ---
    (
        "data processing",
        """\
data = [
    {"name": "alice", "score": 95},
    {"name": "bob", "score": 87},
    {"name": "carol", "score": 92},
]
best = sorted(data, key=lambda x: x["score"], reverse=True)[0]
avg = sum(d["score"] for d in data) / len(data)
print(f'{best["name"]} {best["score"]} {avg:.1f}')
""",
        None,
        "alice 95 91.3",
    ),
    # --- Sorting algorithms ---
    (
        "sorting (quicksort)",
        """\
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + mid + quicksort(right)

nums = [38, 27, 43, 3, 9, 82, 10, 1, 72, 55, 41, 23, 64, 17, 91]
print(quicksort(nums))
""",
        None,
        "[1, 3, 9, 10, 17, 23, 27, 38, 41, 43, 55, 64, 72, 82, 91]",
    ),
    # --- Regex (both support re module now) ---
    (
        "regex",
        """\
import re
text = "Contact us at info@example.com or sales@test.org for details"
emails = re.findall(r'[\\w.+-]+@[\\w-]+\\.[\\w.]+', text)
print(len(emails), emails[0])
""",
        None,
        "2 info@example.com",
    ),
    # --- Math-heavy computation ---
    (
        "math operations",
        """\
import math
results = []
for i in range(1, 20):
    val = math.sqrt(i) + math.log(i + 1) + math.sin(i)
    results.append(round(val, 4))
print(results[0], results[9], results[18])
""",
        None,
        None,
    ),
]

# --- State accumulation tasks (run sequentially on same interpreter) ---

STATE_TASKS: list[tuple[str, str, dict[str, Any] | None]] = [
    ("state:define", "counter = 0\ndata = []", None),
    ("state:mutate1", "counter += 10\ndata.append(counter)", None),
    ("state:mutate2", "counter += 20\ndata.append(counter)", None),
    ("state:mutate3", "counter += 30\ndata.append(counter)", None),
    ("state:mutate4", "counter += 40\ndata.append(counter)", None),
    ("state:read", "print(counter, data)", None),
]

# --- Tool call tasks ---

TOOL_TASKS: list[tuple[str, str]] = [
    ("tool:single", 'result = lookup(key="test")\nprint(result)'),
    ("tool:multi", 'a = lookup(key="x")\nb = lookup(key="y")\nprint(a + " " + b)'),
    ("tool:in-loop", 'results = []\nfor k in ["a", "b", "c", "d", "e"]:\n    results.append(lookup(key=k))\nprint(len(results))'),
    ("tool:with-processing", 'raw = lookup(key="data")\nwords = raw.split("_")\nprint("-".join(w.upper() for w in words))'),
]

# --- SUBMIT tasks ---

SUBMIT_TASKS: list[tuple[str, str, list[dict]]] = [
    ("submit:kwargs", 'SUBMIT(answer="hello", confidence=0.95)',
     [{"name": "answer", "type": "str"}, {"name": "confidence", "type": "float"}]),
    ("submit:computed", "x = 21 * 2\nSUBMIT(result=x)",
     [{"name": "result", "type": "int"}]),
]


# ---------------------------------------------------------------------------
# Timing infrastructure
# ---------------------------------------------------------------------------


@dataclass
class TaskResult:
    name: str
    time_ms: float
    output: str | None = None
    error: str | None = None
    correct: bool = True


@dataclass
class BenchmarkRun:
    interpreter: str
    startup_ms: float
    shutdown_ms: float
    tasks: list[TaskResult] = field(default_factory=list)

    @property
    def total_exec_ms(self) -> float:
        return sum(t.time_ms for t in self.tasks)

    @property
    def total_ms(self) -> float:
        return self.startup_ms + self.total_exec_ms + self.shutdown_ms

    @property
    def pass_count(self) -> int:
        return sum(1 for t in self.tasks if t.error is None)

    @property
    def fail_count(self) -> int:
        return sum(1 for t in self.tasks if t.error is not None)


def _run_task(interp: Any, code: str, variables: dict | None) -> tuple[float, str | None, str | None]:
    """Execute code and return (time_ms, output_str, error_str)."""
    t0 = time.perf_counter()
    try:
        result = interp.execute(code, variables=variables)
        elapsed = (time.perf_counter() - t0) * 1000
        if isinstance(result, FinalOutput):
            return elapsed, f"FinalOutput({result.output})", None
        return elapsed, str(result) if result is not None else None, None
    except (CodeInterpreterError, SyntaxError) as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return elapsed, None, f"{type(e).__name__}: {e}"


def run_benchmark(interpreter_cls: type, label: str, tools: dict | None = None,
                  output_fields: list[dict] | None = None) -> BenchmarkRun:
    """Run the full benchmark suite on a given interpreter class."""

    # --- Startup ---
    t0 = time.perf_counter()
    if interpreter_cls is PythonInterpreter:
        interp = interpreter_cls(tools=tools or {}, output_fields=output_fields)
        interp.start()
        # Force warm-up: PythonInterpreter lazily starts Deno on first execute
        interp.execute("1 + 1")
    else:
        interp = interpreter_cls(tools=tools or {}, output_fields=output_fields)
        interp.start()
        interp.execute("1 + 1")
    startup_ms = (time.perf_counter() - t0) * 1000

    run = BenchmarkRun(interpreter=label, startup_ms=startup_ms, shutdown_ms=0)

    # --- Independent tasks (fresh interpreter each time for fair comparison) ---
    for name, code, variables, expected in TASKS:
        # Create fresh interpreter for each independent task
        if interpreter_cls is PythonInterpreter:
            task_interp = interpreter_cls(tools=tools or {}, output_fields=output_fields)
        else:
            task_interp = interpreter_cls(tools=tools or {}, output_fields=output_fields)

        elapsed, output, error = _run_task(task_interp, code, variables)
        correct = True
        if expected and output and expected not in output:
            correct = False
        run.tasks.append(TaskResult(name=name, time_ms=elapsed, output=output, error=error, correct=correct))

        if interpreter_cls is PythonInterpreter:
            task_interp.shutdown()

    # --- State accumulation tasks (same interpreter) ---
    if interpreter_cls is PythonInterpreter:
        state_interp = interpreter_cls(tools=tools or {}, output_fields=output_fields)
    else:
        state_interp = interpreter_cls(tools=tools or {}, output_fields=output_fields)

    for name, code, variables in STATE_TASKS:
        elapsed, output, error = _run_task(state_interp, code, variables)
        run.tasks.append(TaskResult(name=name, time_ms=elapsed, output=output, error=error))

    if interpreter_cls is PythonInterpreter:
        state_interp.shutdown()

    # --- Tool call tasks ---
    def lookup(key: str) -> str:
        return f"value_for_{key}"

    tool_dict = {"lookup": lookup}
    if interpreter_cls is PythonInterpreter:
        tool_interp = interpreter_cls(tools=tool_dict, output_fields=output_fields)
    else:
        tool_interp = interpreter_cls(tools=tool_dict, output_fields=output_fields)

    for name, code in TOOL_TASKS:
        elapsed, output, error = _run_task(tool_interp, code, None)
        run.tasks.append(TaskResult(name=name, time_ms=elapsed, output=output, error=error))

    if interpreter_cls is PythonInterpreter:
        tool_interp.shutdown()

    # --- SUBMIT tasks (each with matching output_fields) ---
    for name, code, fields in SUBMIT_TASKS:
        submit_interp = interpreter_cls(tools={}, output_fields=fields)
        elapsed, output, error = _run_task(submit_interp, code, None)
        run.tasks.append(TaskResult(name=name, time_ms=elapsed, output=output, error=error))
        if interpreter_cls is PythonInterpreter:
            submit_interp.shutdown()

    # --- Shutdown (main warmed interpreter) ---
    t0 = time.perf_counter()
    if interpreter_cls is PythonInterpreter:
        interp.shutdown()
    else:
        interp.shutdown()
    run.shutdown_ms = (time.perf_counter() - t0) * 1000

    return run


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

CYAN = "\033[36m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def print_run(run: BenchmarkRun) -> None:
    print(f"\n{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}  {run.interpreter}{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")
    print(f"  {'Startup:':<22} {CYAN}{run.startup_ms:>10.2f} ms{RESET}")
    print(f"  {'Shutdown:':<22} {CYAN}{run.shutdown_ms:>10.2f} ms{RESET}")
    print(f"  {'Total execution:':<22} {CYAN}{run.total_exec_ms:>10.2f} ms{RESET}")
    print(f"  {'Total wall clock:':<22} {BOLD}{run.total_ms:>10.2f} ms{RESET}")
    print(f"  {'Tasks passed:':<22} {GREEN}{run.pass_count}{RESET} / {len(run.tasks)}")
    if run.fail_count:
        print(f"  {'Tasks failed:':<22} {RED}{run.fail_count}{RESET}")
    print()

    # Group tasks
    independent = [t for t in run.tasks if not t.name.startswith(("state:", "tool:", "submit:"))]
    state = [t for t in run.tasks if t.name.startswith("state:")]
    tools = [t for t in run.tasks if t.name.startswith("tool:")]
    submits = [t for t in run.tasks if t.name.startswith("submit:")]

    def print_section(title: str, tasks: list[TaskResult]) -> None:
        if not tasks:
            return
        print(f"  {BOLD}{title}{RESET}")
        for t in tasks:
            status = f"{GREEN}OK{RESET}" if t.error is None else f"{RED}ERR{RESET}"
            correct = "" if t.correct else f" {YELLOW}(wrong output){RESET}"
            print(f"    {t.name:<32} {t.time_ms:>8.2f} ms  [{status}]{correct}")
            if t.error:
                short = t.error[:80] + ("..." if len(t.error) > 80 else "")
                print(f"    {DIM}  {short}{RESET}")
        section_total = sum(t.time_ms for t in tasks)
        print(f"    {'subtotal':<32} {BOLD}{section_total:>8.2f} ms{RESET}")
        print()

    print_section("Independent Tasks", independent)
    print_section("State Accumulation (sequential)", state)
    print_section("Tool Calls", tools)
    print_section("SUBMIT", submits)


def print_comparison(monty: BenchmarkRun, python: BenchmarkRun) -> None:
    print(f"\n{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}  COMPARISON: Monty vs PythonInterpreter{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}\n")

    def ratio_str(monty_val: float, python_val: float) -> str:
        if monty_val == 0 and python_val == 0:
            return "—"
        if monty_val == 0:
            return f"{GREEN}∞x faster{RESET}"
        r = python_val / monty_val
        if r >= 1:
            return f"{GREEN}{r:.1f}x faster{RESET}"
        else:
            return f"{RED}{1/r:.1f}x slower{RESET}"

    rows = [
        ("Startup", monty.startup_ms, python.startup_ms),
        ("Shutdown", monty.shutdown_ms, python.shutdown_ms),
        ("Total execution", monty.total_exec_ms, python.total_exec_ms),
        ("Total wall clock", monty.total_ms, python.total_ms),
    ]

    print(f"  {'Metric':<24} {'Monty':>10}  {'Python':>10}  {'Monty advantage':>18}")
    print(f"  {'-' * 66}")
    for label, m_val, p_val in rows:
        print(f"  {label:<24} {m_val:>8.2f}ms  {p_val:>8.2f}ms  {ratio_str(m_val, p_val):>30}")

    # Per-task comparison
    print(f"\n  {BOLD}Per-task breakdown:{RESET}")
    print(f"  {'Task':<34} {'Monty':>8}  {'Python':>8}  {'Speedup':>12}")
    print(f"  {'-' * 66}")
    for mt, pt in zip(monty.tasks, python.tasks, strict=True):
        r = ratio_str(mt.time_ms, pt.time_ms)
        m_status = "   " if mt.error is None else f"{RED}ERR{RESET}"
        p_status = "   " if pt.error is None else f"{RED}ERR{RESET}"
        print(f"  {mt.name:<34} {mt.time_ms:>6.2f}ms  {pt.time_ms:>6.2f}ms  {r:>24}  {m_status} {p_status}")


def print_multi_run_summary(monty_runs: list[BenchmarkRun], python_runs: list[BenchmarkRun]) -> None:
    n = len(monty_runs)
    print(f"\n{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}  SUMMARY ({n} runs, median values){RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}\n")

    def med(vals: list[float]) -> float:
        return statistics.median(vals)

    m_startup = med([r.startup_ms for r in monty_runs])
    m_exec = med([r.total_exec_ms for r in monty_runs])
    m_total = med([r.total_ms for r in monty_runs])
    p_startup = med([r.startup_ms for r in python_runs])
    p_exec = med([r.total_exec_ms for r in python_runs])
    p_total = med([r.total_ms for r in python_runs])

    def ratio_str(m: float, p: float) -> str:
        if m == 0:
            return f"{GREEN}∞x{RESET}"
        r = p / m
        color = GREEN if r >= 1 else RED
        label = "faster" if r >= 1 else "slower"
        return f"{color}{r:.1f}x {label}{RESET}"

    print(f"  {'Metric':<24} {'Monty':>10}  {'Python':>10}  {'Monty advantage':>18}")
    print(f"  {'-' * 66}")
    print(f"  {'Startup (median)':<24} {m_startup:>8.2f}ms  {p_startup:>8.2f}ms  {ratio_str(m_startup, p_startup):>30}")
    print(f"  {'Execution (median)':<24} {m_exec:>8.2f}ms  {p_exec:>8.2f}ms  {ratio_str(m_exec, p_exec):>30}")
    print(f"  {'Wall clock (median)':<24} {m_total:>8.2f}ms  {p_total:>8.2f}ms  {ratio_str(m_total, p_total):>30}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MontyInterpreter vs PythonInterpreter")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs to average (default: 3)")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    parser.add_argument("--monty-only", action="store_true", help="Only run Monty benchmark")
    parser.add_argument("--python-only", action="store_true", help="Only run Python benchmark")
    args = parser.parse_args()

    run_monty = not args.python_only
    run_python = not args.monty_only

    monty_runs: list[BenchmarkRun] = []
    python_runs: list[BenchmarkRun] = []

    for i in range(args.runs):
        if not args.json:
            print(f"\n{DIM}--- Run {i + 1}/{args.runs} ---{RESET}")

        if run_monty:
            if not args.json:
                print(f"{DIM}  Running MontyInterpreter...{RESET}", end="", flush=True)
            monty_runs.append(run_benchmark(MontyInterpreter, "MontyInterpreter"))
            if not args.json:
                print(f" {GREEN}done{RESET} ({monty_runs[-1].total_ms:.0f}ms)")

        if run_python:
            if not args.json:
                print(f"{DIM}  Running PythonInterpreter...{RESET}", end="", flush=True)
            python_runs.append(run_benchmark(PythonInterpreter, "PythonInterpreter"))
            if not args.json:
                print(f" {GREEN}done{RESET} ({python_runs[-1].total_ms:.0f}ms)")

    if args.json:
        out = {
            "runs": args.runs,
            "monty": [{
                "startup_ms": r.startup_ms,
                "shutdown_ms": r.shutdown_ms,
                "total_exec_ms": r.total_exec_ms,
                "total_ms": r.total_ms,
                "pass_count": r.pass_count,
                "fail_count": r.fail_count,
                "tasks": [{"name": t.name, "time_ms": t.time_ms, "error": t.error} for t in r.tasks],
            } for r in monty_runs],
            "python": [{
                "startup_ms": r.startup_ms,
                "shutdown_ms": r.shutdown_ms,
                "total_exec_ms": r.total_exec_ms,
                "total_ms": r.total_ms,
                "pass_count": r.pass_count,
                "fail_count": r.fail_count,
                "tasks": [{"name": t.name, "time_ms": t.time_ms, "error": t.error} for t in r.tasks],
            } for r in python_runs],
        }
        print(json.dumps(out, indent=2))
        return

    # Print best run for each
    if monty_runs:
        best_monty = min(monty_runs, key=lambda r: r.total_exec_ms)
        print_run(best_monty)

    if python_runs:
        best_python = min(python_runs, key=lambda r: r.total_exec_ms)
        print_run(best_python)

    if monty_runs and python_runs:
        print_comparison(best_monty, best_python)

    if args.runs > 1 and monty_runs and python_runs:
        print_multi_run_summary(monty_runs, python_runs)

    print()


if __name__ == "__main__":
    main()
