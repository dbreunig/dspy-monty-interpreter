# dspy-monty-interpreter

DSPy `CodeInterpreter` implementation using [Monty](https://github.com/pydantic/monty), a secure Python interpreter written in Rust.

The Monty team points out, "This project is still in development, and not ready for the prime time." It uses a small subset of the standard library (`sys`, `os`, `typing`, `asyncio`, `re`, `datetime`, `json`, `math`) and can't yet define classes or use match statements. It does support `with`/context managers and a sandboxed `open()` (file access is opt-in via the `mounts` and `os_access` parameters).

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

### Filesystem access

Monty's sandbox can read and write files when you opt in. Mount a host
directory with `MountDir`, or supply an in-memory filesystem via `os_access`:

```python
from dspy_monty_interpreter import MontyInterpreter, MountDir

# read-only host dir; "overlay" keeps writes in memory, leaving the host alone
interp = MontyInterpreter(mounts=MountDir("/data", "/path/on/host", mode="read-only"))
interp.execute("from pathlib import Path\nprint(Path('/data/notes.txt').read_text())")
```

## Examples

See [`examples/`](examples/) for patterns built on top of the library:

- [`examples/repo_rlm.py`](examples/repo_rlm.py) — `RepoRLM`, a `dspy.RLM`
  specialized for analyzing one or more repositories. It mounts repos (local
  checkouts or cloned from `owner/repo`) into Monty and surfaces them to the LLM
  via a prompt manifest plus a `repos` REPL variable, so the model can read the
  code and produce a report.

## Why Monty?

- **Fast**: Microsecond startup (no subprocess, no WASM bootstrap)
- **Secure**: No filesystem, network, or environment access by default
- **Lightweight**: Pure Rust, no Deno/Pyodide dependency
