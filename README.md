# dspy-monty-interpreter

DSPy `CodeInterpreter` implementation using [Monty](https://github.com/pydantic/monty), a secure Python interpreter written in Rust.

Monty describes itself as "a sandbox for Python code written by AI." It implements a subset of the standard library (`sys`, `os`, `pathlib`, `typing`, `asyncio`, `re`, `datetime`, `time`, `random`, `json`, `math`, `unicodedata`, `collections`, `itertools`, `functools`, `dataclasses`, `copy`, `base64`, `binascii`) and can't yet use match statements, generator functions (`yield`), or class inheritance. It does support classes, decorators, `@dataclass` (including `frozen=` and `eq=`), generator expressions, `with`/context managers, and a sandboxed `open()` (file access is opt-in, see [Filesystem access](#filesystem-access)).

That said: Monty is *fast*. For many RLM use cases, Monty is my daily driver.

## Installation

```bash
pip install dspy-monty-interpreter
```

Requires `dspy>=3.3.1` and `pydantic-monty>=1.0.0`. Since 1.0, `pydantic-monty` is a metapackage that installs `pydantic-monty-client` (the Python bindings) and `pydantic-monty-runtime` (the `monty` worker binary). If the binary comes from somewhere else, such as a base image, install `pydantic-monty-client` alone and point it at the binary with the `MONTY_BIN` environment variable.

## Usage

```python
import dspy
from dspy_monty_interpreter import MontyInterpreter

rlm = dspy.RLM("context -> answer", interpreter_factory=MontyInterpreter)
result = rlm(context="What is 2 + 2?")
```

DSPy calls `interpreter_factory` once per `forward()` and shuts the interpreter down afterwards. To configure the interpreter, use `MontyInterpreter.factory(...)`, which takes the same arguments as the constructor:

```python
rlm = dspy.RLM(
    "context -> answer",
    interpreter_factory=MontyInterpreter.factory(request_timeout=10.0),
)
```

To reuse one interpreter across calls instead (DSPy 3.2's `interpreter=` constructor argument), pass it an instance of `MontyInterpreter` as the first positional argument when calling the module. RLM injects its tools into it but never shuts it down, which is the same contract as before; only the call site moved:

```python
interpreter = MontyInterpreter(request_timeout=10.0)
result = rlm(interpreter, context="What is 2 + 2?")
```

Keep `interpreter_factory=MontyInterpreter` on the RLM even when you pass an interpreter positionally. RLM builds its action prompt once, at construction time, from the factory's `execution_instructions`; the interpreter you pass at call time is too late to affect it.

## Filesystem access

The sandbox has no filesystem access by default. You grant access per interpreter by passing one or more `MountDir` objects, which map a virtual path inside the sandbox to a directory on your machine. Sandboxed code then reads and writes those paths with the normal `open()` and `pathlib.Path` APIs.

A `MountDir` takes a virtual path, a host path, and a mode (all keyword-only):

- `read-only`: code can read the files but cannot write anything.
- `read-write`: code can read and write the real files on your machine.
- `overlay`: reads come from your real files, but writes are kept in memory for the duration of one `execute()` call and never touch your disk.

For a typical agent, mount the files you want the model to explore as read-only, and add an overlay directory for anything it wants to write:

```python
import dspy
from dspy_monty_interpreter import MontyInterpreter, MountDir

factory = MontyInterpreter.factory(mounts=[
    MountDir(virtual_path="/data", host_path="./reports", mode="read-only"),
    MountDir(virtual_path="/scratch", host_path="./scratch", mode="overlay"),
])
rlm = dspy.RLM("question -> answer", interpreter_factory=factory)
result = rlm(question="Which report mentions the Q3 forecast?")
```

The model can read every file under `./reports` through `/data`, and it can write freely under `/scratch` during each `execute()` call, but nothing it writes ever reaches your disk.

Because the factory knows about the mounts, its `execution_instructions` tell the model what is there. For the example above the prompt gains a section like:

```
Mounted directories (explore with os.listdir(path), pathlib.Path(path).iterdir(), and open(path); os.path and os.walk are not available):
- /data (read-only) containing q3/, readme.md
- /scratch (overlay: writable, but writes are discarded when each execution ends) containing (empty)
```

The sandbox's working directory starts at the first mount's virtual path (`/data` above, or `/` without mounts), so relative paths like `open('q3/summary.md')` resolve inside it, and `os.chdir()` persists across `execute()` calls.

The listing is a snapshot of each host directory's top level at the time the factory is created, and it is bounded so a large mount cannot bloat the prompt. Up to 20 entries (`mount_listing_limit`) are listed by name; beyond that the model gets a summary instead, for example `1,204 files and 1 directory (.csv ×1200, .md ×4; e.g. 0000.csv, 0001.csv, 0002.csv)`. Individual names are cut at 40 characters, each mount's description at roughly 300, and the directory scan stops at 5,000 entries. Pass `mount_listing_limit=0` to describe only the path and mode. Without mounts the section says the filesystem is unavailable, so the model does not go looking for one. To reuse one interpreter across calls, pass it positionally (`rlm(factory(), question=...)`); the instance carries the same `execution_instructions` if you need them elsewhere.

Overlay mode also works well as pure scratch space in standalone use:

```python
import tempfile
from dspy_monty_interpreter import MontyInterpreter, MountDir

interp = MontyInterpreter(
    mounts=MountDir(virtual_path="/data", host_path=tempfile.mkdtemp(), mode="overlay")
)

interp.execute(
    "from pathlib import Path\n"
    "Path('/data/notes.txt').write_text('draft')\n"
    "print(Path('/data/notes.txt').read_text())"
)  # returns "draft"
# The host directory is still empty.
```

Two things to know about overlays. First, an overlay only lives for the duration of a single `execute()` call. Code that writes a file must read it back in the same call. Use `read-write` mode when files need to survive across calls. Second, overlay writes never reach the host, so the only way to get overlay data out is from inside the sandbox. Have the code `print()` or `SUBMIT()` the results, or use `read-write` mode when you need real files on disk.

For a fully virtual filesystem or environment variables, pass an `AbstractOS` implementation (such as `pydantic_monty.OSAccess`) as the `os_access` parameter. See the [Monty documentation](https://github.com/pydantic/monty) for details. To control the clock, see [Clock, sleep, and randomness](#clock-sleep-and-randomness).

## Clock, sleep, and randomness

The sandbox answers `datetime.now()`, `date.today()`, `time.time()`, and `random` from the worker's own clock and entropy, in UTC. Pass an `OSPolicy` dict as `os_policy` to change that, for example to freeze the clock and seed `random` so an RLM run is reproducible:

```python
from datetime import datetime

interp = MontyInterpreter(
    os_policy={
        "datetime": datetime(2026, 1, 1, 12, 0),
        "timezone": "America/Los_Angeles",
        "random_start": {"seed": 42},
    }
)
```

`MontyInterpreter` sets `sleep` to `"zero"` by default, so `time.sleep()` and `asyncio.sleep()` return immediately instead of holding a worker (Monty otherwise waits, capped at ten seconds per call). Pass `os_policy={"sleep": "system"}` to restore real waits. Keys you leave out keep this default and Monty's.

An `os_access` object's `datetime_now()` and `date_today()` overrides are only consulted when `os_policy` routes the clock to the host: `os_policy={"datetime": "call_host"}`. Before Monty 1.0 they were called unconditionally, so upgrade any code that relied on that.

## Resource limits

Pass a `ResourceLimits` dict to cap what each `execute()` call may consume. The most useful knob is `max_memory` (bytes): the interpreter's allocator enforces it, so a runaway allocation raises a `MemoryError` inside the sandbox instead of exhausting the host.

```python
from pydantic_monty import ResourceLimits

interp = MontyInterpreter(
    resource_limits=ResourceLimits(max_memory=256 * 1024 * 1024)
)
```

Two keys cap execution time: `max_feed_duration_secs` limits one `execute()` call, and `max_turn_duration_secs` limits the sandbox time between tool calls. Both count only time spent running sandboxed code, not time waiting on your tools. Other keys are `max_recursion_depth` and `max_suspensions` (both default to 1000 and cannot be disabled), `max_total_sleep_secs`, and `gc_interval`. Monty rejects unknown keys with a `ValueError` on the first `execute()`.

A limit violation surfaces as a `CodeExecutionError` from `execute()`, which RLM feeds back to the model. After a memory, recursion, or suspension limit the session keeps its state. After a time limit Monty makes no guarantees about the sandbox heap, so the adapter discards the session: the next `execute()` starts with empty state, and the error message tells the model so.

## Timeouts

Pass `request_timeout` to set a hard limit, in seconds, on each `execute()` call:

```python
interp = MontyInterpreter(request_timeout=10.0)
```

When code exceeds the limit, `execute()` raises `CodeInterpreterError`. The session state is lost, and the next `execute()` starts fresh. (Ordinary runtime errors in the code raise `CodeExecutionError`, a subclass that RLM feeds back to the model as a correction turn; a timeout is terminal and ends the RLM run.)

## Parallel evaluation

A single `MontyInterpreter` is safe to share across threads, which is exactly what `dspy.Evaluate` and `dspy.Parallel` do when they run one RLM with `num_threads`. Each thread gets its own isolated REPL session, and all sessions share one pool of Monty worker processes:

```python
interpreter = MontyInterpreter(max_processes=8)
rlm = dspy.RLM("question -> answer", interpreter_factory=MontyInterpreter)
program = lambda **inputs: rlm(interpreter, **inputs)
evaluate = dspy.Evaluate(devset=devset, num_threads=8, metric=metric)
evaluate(program)
```

Passing the interpreter positionally is what shares the pool. With `interpreter_factory` alone, every `forward()` spins up and tears down its own worker pool, which works but is slower. DSPy documents a caller-owned interpreter as safe only for sequential calls; `MontyInterpreter`'s per-thread sessions are what make the concurrent case work, and the e2e suite exercises it.

The pool caps live workers at `max_processes`, which defaults to your CPU count. A thread holds its worker between `execute()` calls, so when `num_threads` is higher than your CPU count, set `max_processes` to at least `num_threads`. Otherwise threads at the tail of a run can wait on workers that idle threads still hold.

## Observability

`MontyInterpreter` fires DSPy's interpreter lifecycle callbacks: `on_interpreter_startup_*`, `on_interpreter_execute_*`, `on_interpreter_tool_call_*`, and `on_interpreter_shutdown_*`. Register a `dspy.utils.callback.BaseCallback` globally with `dspy.configure(callbacks=[...])` or per instance with `MontyInterpreter(callbacks=[...])`. Tool calls nest under the `execute()` call that made them, so traces keep their ancestry.

## Why Monty?

- **Fast**: No WASM bootstrap. Code runs against a pool of warm `monty` worker processes
- **Secure**: No filesystem, network, or environment access unless you grant it (see [Filesystem access](#filesystem-access))
- **Resilient**: A crashed or timed-out worker is replaced automatically without taking down your process
- **Lightweight**: Pure Rust, no Deno/Pyodide dependency
