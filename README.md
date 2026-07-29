# dspy-monty-interpreter

DSPy `CodeInterpreter` implementation using [Monty](https://github.com/pydantic/monty), a secure Python interpreter written in Rust.

The Monty team points out, "This project is still in development, and not ready for the prime time." It uses a small subset of the standard library (`sys`, `os`, `typing`, `asyncio`, `re`, `datetime`, `json`, `math`, `unicodedata`) and can't yet use match statements. It does support classes (as of Monty 0.0.19), `with`/context managers, and a sandboxed `open()` (file access is opt-in, see [Filesystem access](#filesystem-access)).

That said: Monty is *fast*. For many RLM use cases, Monty is my daily driver.

## Installation

```bash
pip install dspy-monty-interpreter
```

Requires `pydantic-monty>=0.0.19`.

## Usage

```python
import dspy
from dspy_monty_interpreter import MontyInterpreter

interpreter = MontyInterpreter()
rlm = dspy.RLM("context -> answer", interpreter=interpreter)
result = rlm(context="What is 2 + 2?")
```

### Standalone usage

```python
from dspy_monty_interpreter import MontyInterpreter

interp = MontyInterpreter()

# Basic execution
interp.execute("x = 42")
interp.execute("print(x + 8)")  # returns "50"

# State persists across calls
interp.execute("def double(n):\n    return n * 2")
interp.execute("double(21)")  # returns "42"

# With tools
def lookup(key: str) -> str:
    return "some value"

interp = MontyInterpreter(tools={"lookup": lookup})
interp.execute('result = lookup(key="foo")\nprint(result)')
```

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

interpreter = MontyInterpreter(mounts=[
    MountDir(virtual_path="/data", host_path="./reports", mode="read-only"),
    MountDir(virtual_path="/scratch", host_path="./scratch", mode="overlay"),
])
rlm = dspy.RLM("question -> answer", interpreter=interpreter)
result = rlm(question="Which report mentions the Q3 forecast?")
```

The model can read every file under `./reports` through `/data`, and it can write freely under `/scratch` during each `execute()` call, but nothing it writes ever reaches your disk.

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

Two things to know about overlays. First, as of Monty 0.0.19 an overlay only lives for the duration of a single `execute()` call. Code that writes a file must read it back in the same call. Use `read-write` mode when files need to survive across calls. Second, overlay writes never reach the host, so the only way to get overlay data out is from inside the sandbox. Have the code `print()` or `SUBMIT()` the results, or use `read-write` mode when you need real files on disk.

For a fully virtual filesystem, environment variables, or control over the clock, pass an `AbstractOS` implementation as the `os_access` parameter. See the [Monty documentation](https://github.com/pydantic/monty) for details.

## Timeouts

Pass `request_timeout` to set a hard limit, in seconds, on each `execute()` call:

```python
interp = MontyInterpreter(request_timeout=10.0)
```

When code exceeds the limit, `execute()` raises `CodeInterpreterError`. The session state is lost, and the next `execute()` starts fresh.

## Parallel evaluation

A single `MontyInterpreter` is safe to share across threads, which is exactly what `dspy.Evaluate` and `dspy.Parallel` do when they run one RLM with `num_threads`. Each thread gets its own isolated REPL session, and all sessions share one pool of Monty worker processes:

```python
interpreter = MontyInterpreter(max_processes=8)
rlm = dspy.RLM("question -> answer", interpreter=interpreter)
evaluate = dspy.Evaluate(devset=devset, num_threads=8, metric=metric)
evaluate(rlm)
```

The pool caps live workers at `max_processes`, which defaults to your CPU count. A thread holds its worker between `execute()` calls, so when `num_threads` is higher than your CPU count, set `max_processes` to at least `num_threads`. Otherwise threads at the tail of a run can wait on workers that idle threads still hold.

## Why Monty?

- **Fast**: No WASM bootstrap. Code runs against a pool of warm `monty` worker processes (as of Monty 0.0.19)
- **Secure**: No filesystem, network, or environment access unless you grant it (see [Filesystem access](#filesystem-access))
- **Resilient**: A crashed or timed-out worker is replaced automatically without taking down your process
- **Lightweight**: Pure Rust, no Deno/Pyodide dependency
