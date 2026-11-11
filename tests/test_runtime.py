"""Monty runtime behavior: threads, resource limits, crashes, stdlib, and os_policy."""

import pytest
from dspy.primitives.code_interpreter import CodeInterpreterError

from dspy_monty_interpreter import MontyInterpreter, MountDir

# --- pydantic-monty 0.0.18 regressions ---


def test_comprehension_store_in_repl():
    """REPL store of names bound inside comprehension expressions (monty #297).
    Previously, building a comprehension that referenced persisted state and
    assigning the result could fail under MontyRepl."""
    interp = MontyInterpreter()
    interp.execute("nums = [1, 2, 3, 4]")
    assert interp.execute("squares = [n * n for n in nums]\nprint(sum(squares))") == "30"
    # The comprehension result persists and is reusable on the next call.
    assert interp.execute("len(squares)") == "4"
    # Dict comprehension referencing prior state behaves the same.
    assert interp.execute("m = {n: n * n for n in nums}\nprint(m[3])") == "9"


def test_context_manager_with_open():
    """`with` / context-manager support plus the sandboxed open() builtin
    (monty #462, #456, #461), exercised over an overlay mount. Overlay
    writes are per-execute, so the write/read round-trip happens in one
    execute() call."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "in.txt").write_text("line one\nline two\n")
        interp = MontyInterpreter(
            mounts=MountDir(virtual_path="/data", host_path=tmpdir, mode="overlay")
        )
        assert interp.execute(
            "with open('/data/in.txt') as f:\n    data = f.read()\nprint(len(data))"
        ) == "18"
        assert interp.execute(
            "with open('/data/out.txt', 'w') as f:\n    f.write('hello cm')\n"
            "with open('/data/out.txt') as f:\n    print(f.read())"
        ) == "hello cm"


def test_parallel_execute_thread_isolation():
    """Concurrent execute() calls from different threads get isolated
    sessions: each thread's variables are its own."""
    import threading

    interp = MontyInterpreter()
    barrier = threading.Barrier(2)
    results: dict[str, str] = {}
    errors: list[Exception] = []

    def worker(name: str, value: int) -> None:
        try:
            barrier.wait()
            interp.execute(f"x = {value}")
            barrier.wait()
            results[name] = interp.execute("print(x)")
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=worker, args=("a", 1)),
        threading.Thread(target=worker, args=("b", 2)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert results == {"a": "1", "b": "2"}
    interp.shutdown()


def test_parallel_reset_only_clears_calling_thread():
    """RLM's per-forward reset (_tools_registered = False) discards only
    the calling thread's session, not other threads' state."""
    import threading

    interp = MontyInterpreter()
    interp.execute("x = 'main'")

    def other_forward() -> None:
        # Simulates RLM starting a forward() on another thread.
        interp._tools_registered = False
        interp.execute("x = 'other'")

    t = threading.Thread(target=other_forward)
    t.start()
    t.join()

    # Main thread's state survived the other thread's reset.
    assert interp.execute("print(x)") == "main"
    interp.shutdown()


def test_parallel_evaluate_smoke():
    """Many sequential tasks spread over a thread pool, each doing an
    RLM-style reset + stateful execute sequence, all come back correct."""
    from concurrent.futures import ThreadPoolExecutor

    interp = MontyInterpreter(max_processes=4)

    def task(i: int) -> str:
        interp._tools_registered = False  # RLM does this per forward()
        interp.execute(f"v = {i}")
        interp.execute("v = v * 10")
        return interp.execute("print(v)")

    with ThreadPoolExecutor(max_workers=4) as ex:
        outputs = list(ex.map(task, range(12)))

    assert outputs == [str(i * 10) for i in range(12)]
    interp.shutdown()


def test_shutdown_reclaims_all_thread_sessions():
    """shutdown() discards sessions created by other threads, and the
    interpreter remains usable afterward with fresh state."""
    import threading

    interp = MontyInterpreter()
    ready = threading.Event()
    release = threading.Event()

    def worker() -> None:
        interp.execute("y = 'thread'")
        ready.set()
        release.wait()

    t = threading.Thread(target=worker)
    t.start()
    ready.wait()
    interp.execute("y = 'main'")
    interp.shutdown()
    release.set()
    t.join()

    # Fresh session after shutdown: y is gone.
    with pytest.raises(CodeInterpreterError):
        interp.execute("y")
    interp.shutdown()


def test_user_defined_class():
    """User-defined classes work as of Monty 0.0.19, including state
    that persists across execute() calls."""
    interp = MontyInterpreter()
    interp.execute(
        "class Counter:\n"
        "    def __init__(self):\n"
        "        self.n = 0\n"
        "    def bump(self):\n"
        "        self.n += 1\n"
        "        return self.n\n"
        "c = Counter()"
    )
    assert interp.execute("c.bump()\nc.bump()\nprint(c.bump())") == "3"


def test_max_memory_limit_raises_and_keeps_state():
    """A max_memory violation surfaces as CodeInterpreterError (MemoryError
    in the sandbox) and, unlike a worker crash, the session keeps its state."""
    from pydantic_monty import ResourceLimits

    interp = MontyInterpreter(resource_limits=ResourceLimits(max_memory=50_000_000))
    interp.start()
    try:
        interp.execute("x = 42")
        with pytest.raises(CodeInterpreterError, match="MemoryError"):
            interp.execute("y = [0] * 100_000_000")
        assert interp.execute("x") == "42"
    finally:
        interp.shutdown()


def test_feed_time_limit_resets_session():
    """max_feed_duration_secs surfaces as a recoverable CodeExecutionError,
    but Monty makes no guarantees about the heap afterwards, so the session
    is discarded: the next execute() starts with empty state."""
    from dspy.primitives.code_interpreter import CodeExecutionError
    from pydantic_monty import ResourceLimits

    interp = MontyInterpreter(resource_limits=ResourceLimits(max_feed_duration_secs=0.2))
    interp.execute("x = 42")
    with pytest.raises(CodeExecutionError, match="time limit exceeded.*variables were lost"):
        interp.execute("i = 0\nwhile True:\n    i += 1")
    with pytest.raises(CodeExecutionError, match="NameError"):
        interp.execute("x")
    assert interp.execute("1 + 1") == "2"
    interp.shutdown()


def test_turn_time_limit_resets_session():
    from dspy.primitives.code_interpreter import CodeExecutionError
    from pydantic_monty import ResourceLimits

    interp = MontyInterpreter(resource_limits=ResourceLimits(max_turn_duration_secs=0.2))
    interp.execute("x = 42")
    with pytest.raises(CodeExecutionError, match="turn time limit exceeded"):
        interp.execute("while True:\n    pass")
    with pytest.raises(CodeExecutionError, match="NameError"):
        interp.execute("x")
    interp.shutdown()


def test_user_raised_timeout_error_keeps_state():
    """Only Monty's own limit TimeoutError resets the session; code raising
    TimeoutError itself is an ordinary runtime error."""
    from dspy.primitives.code_interpreter import CodeExecutionError

    interp = MontyInterpreter()
    interp.execute("x = 42")
    with pytest.raises(CodeExecutionError, match="TimeoutError: mine"):
        interp.execute("raise TimeoutError('mine')")
    assert interp.execute("x") == "42"
    interp.shutdown()


def test_worker_crash_raises_and_recovers():
    """A crashed worker (here: exceeding request_timeout) surfaces as
    CodeInterpreterError, and the interpreter recovers with a fresh
    session on the next execute()."""
    interp = MontyInterpreter(request_timeout=0.5)
    interp.execute("x = 1")
    with pytest.raises(CodeInterpreterError):
        interp.execute("i = 0\nwhile True:\n    i += 1")
    # Session was lost with the worker; the next call gets a fresh one.
    result = interp.execute("print('alive')")
    assert result == "alive"


def test_external_function_identity():
    """Name-based identity and equality for external function values
    (monty #458) — covers both injected tools and SUBMIT."""
    def my_tool(x: str) -> str:
        return "v"

    interp = MontyInterpreter(tools={"my_tool": my_tool})
    assert interp.execute("f = my_tool\nprint(f is my_tool)") == "True"
    assert interp.execute("f = my_tool\nprint(f == my_tool)") == "True"
    assert interp.execute("print(SUBMIT is SUBMIT)") == "True"


# --- Monty 1.0 stdlib and os_policy ---


def test_monty_1_0_stdlib_modules_import():
    interp = MontyInterpreter()
    assert interp.execute(
        "import random, time, copy, pathlib, binascii\n"
        "random.seed(1)\n"
        "isinstance(time.time(), float) and copy.deepcopy([1])[0]"
    ) == "1"
    interp.shutdown()


def test_decimal_module_imports():
    interp = MontyInterpreter()
    assert interp.execute(
        "from decimal import Decimal, ROUND_HALF_UP\n"
        "print(Decimal('0.1') + Decimal('0.2'), Decimal('2.675').quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))"
    ) == "0.3 2.68"
    interp.shutdown()


def test_fnmatch_and_annotationlib_modules_import():
    interp = MontyInterpreter()
    assert interp.execute(
        "import fnmatch, annotationlib\n"
        "print(fnmatch.fnmatch('q3.csv', '*.csv'))"
    ) == "True"
    interp.shutdown()


def test_python_file_in_mount_is_importable(tmp_path):
    (tmp_path / "helpers.py").write_text("X = 42\n")
    interp = MontyInterpreter(
        mounts=MountDir(virtual_path="/data", host_path=str(tmp_path), mode="read-only")
    )
    assert interp.execute("import helpers\nprint(helpers.X)") == "42"
    interp.shutdown()


def test_datetime_now_works_without_os_access():
    interp = MontyInterpreter()
    assert interp.execute("import datetime\ndatetime.datetime.now().year >= 2026") == "True"
    interp.shutdown()


def test_sleep_returns_immediately_by_default():
    import time

    interp = MontyInterpreter()
    start = time.monotonic()
    assert interp.execute("import time\ntime.sleep(5)\n'ok'") == "ok"
    assert time.monotonic() - start < 1.0
    interp.shutdown()


def test_os_policy_can_restore_system_sleep():
    import time

    interp = MontyInterpreter(os_policy={"sleep": "system"})
    start = time.monotonic()
    interp.execute("import time\ntime.sleep(0.3)")
    assert time.monotonic() - start >= 0.3
    interp.shutdown()


def test_os_policy_freezes_clock_and_seeds_random():
    from datetime import datetime

    interp = MontyInterpreter(
        os_policy={"datetime": datetime(2020, 2, 3, 4, 5), "random_start": {"seed": 7}}
    )
    assert interp.execute("import datetime\nstr(datetime.datetime.now())") == "2020-02-03 04:05:00"
    first = interp.execute("import random\nrandom.random()")
    interp.shutdown()
    interp = MontyInterpreter(os_policy={"random_start": {"seed": 7}})
    assert interp.execute("import random\nrandom.random()") == first
    interp.shutdown()


def test_os_policy_merges_with_default_sleep_zero():
    interp = MontyInterpreter(os_policy={"timezone": "utc"})
    assert interp._os_policy == {"sleep": "zero", "timezone": "utc"}
    assert MontyInterpreter(os_policy={"sleep": "system"})._os_policy == {"sleep": "system"}


def test_os_access_clock_override_needs_call_host():
    """Since Monty 1.0 an AbstractOS clock override is consulted only when
    os_policy routes the clock to the host."""
    from datetime import datetime

    from pydantic_monty import OSAccess

    class Frozen(OSAccess):
        def datetime_now(self, tz=None):
            return datetime(1999, 1, 1)

    code = "import datetime\ndatetime.datetime.now().year"
    interp = MontyInterpreter(os_access=Frozen())
    assert interp.execute(code) != "1999"
    interp.shutdown()
    interp = MontyInterpreter(os_access=Frozen(), os_policy={"datetime": "call_host"})
    assert interp.execute(code) == "1999"
    interp.shutdown()


def test_factory_passes_os_policy():
    factory = MontyInterpreter.factory(os_policy={"sleep": "system"})
    assert factory()._os_policy == {"sleep": "system"}
