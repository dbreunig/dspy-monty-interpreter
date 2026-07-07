# dspy-monty-interpreter

DSPy `CodeInterpreter` implementation using [Monty](https://github.com/pydantic/monty), a secure Python interpreter written in Rust.

The Monty team points out, "This project is still in development, and not ready for the prime time." It uses a small subset of the standard library (`sys`, `os`, `typing`, `asyncio`, `re`, `datetime`, `json`, `math`) and can't yet define classes or use match statements. It does support `with`/context managers and a sandboxed `open()` (file access is opt-in, see [Filesystem access](#filesystem-access)).

That said: Monty is *fast*. For many RLM use cases, Monty is my daily driver.

## Installation

```bash
pip install dspy-monty-interpreter
```

Requires `pydantic-monty>=0.0.18`.

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

A `MountDir` takes a virtual path, a host path, and a mode:

- `read-only`: code can read the files but cannot write anything.
- `read-write`: code can read and write the real files on your machine.
- `overlay`: reads come from your real files, but writes are kept in memory and never touch your disk.

For a typical agent, mount the files you want the model to explore as read-only, and add an overlay directory for anything it wants to write:

```python
import dspy
from dspy_monty_interpreter import MontyInterpreter, MountDir

interpreter = MontyInterpreter(mounts=[
    MountDir("/data", "./reports", mode="read-only"),
    MountDir("/scratch", "./scratch", mode="overlay"),
])
rlm = dspy.RLM("question -> answer", interpreter=interpreter)
result = rlm(question="Which report mentions the Q3 forecast?")
```

The model can read every file under `./reports` through `/data`, and it can write freely under `/scratch`, but nothing it writes ever reaches your disk.

Overlay mode also works well as pure scratch space in standalone use:

```python
import tempfile
from dspy_monty_interpreter import MontyInterpreter, MountDir

interp = MontyInterpreter(
    mounts=MountDir("/data", tempfile.mkdtemp(), mode="overlay")
)

interp.execute(
    "from pathlib import Path\n"
    "Path('/data/notes.txt').write_text('draft')"
)
interp.execute("print(Path('/data/notes.txt').read_text())")  # returns "draft"
# The host directory is still empty.
```

Two things to know about overlays. First, overlay writes are stored on the `MountDir` object, not on the interpreter, so they persist across `execute()` calls, across RLM runs, and even across `shutdown()`. Create a new `MountDir` when you want a fresh overlay. Second, there is no host-side API for reading overlay contents, so the only way to get overlay data out is from inside the sandbox. Have the code `print()` or `SUBMIT()` the results, or use `read-write` mode when you need real files on disk.

For a fully virtual filesystem, environment variables, or control over the clock, pass an `AbstractOS` implementation as the `os_access` parameter. See the [Monty documentation](https://github.com/pydantic/monty) for details.

## Why Monty?

- **Fast**: Microsecond startup (no subprocess, no WASM bootstrap)
- **Secure**: No filesystem, network, or environment access unless you grant it (see [Filesystem access](#filesystem-access))
- **Lightweight**: Pure Rust, no Deno/Pyodide dependency
