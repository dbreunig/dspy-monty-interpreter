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

### Analyzing repositories with `RepoRLM`

`RepoRLM` is a `dspy.RLM` specialized for exploring one or more repositories.
It mounts each repo into the Monty sandbox (read-only by default) and tells the
model what's available — both as a manifest spliced into the prompt and as a
structured `repos` REPL variable — so the LLM can read files with `pathlib` and
build up an answer iteratively.

```python
import dspy
from dspy_monty_interpreter import RepoRLM

dspy.configure(lm=dspy.LM("anthropic/claude-opus-4-8"))

# Local checkout
analyzer = RepoRLM("/home/user/checkouts/requests")
print(analyzer(task="Map the public API, module layout, and key abstractions.").report)
```

```python
# Clone from GitHub on the fly ("owner/repo" or a clone URL).
# Clones persist on disk unless cleanup=True is passed.
analyzer = RepoRLM("psf/requests", cleanup=True)
print(analyzer(task="Summarize the architecture.").report)
```

```python
# Compare multiple repos (names taken from the dict keys)
analyzer = RepoRLM({"requests": "/repos/requests", "httpx": "/repos/httpx"})
analyzer(task="Compare the public client API and dependency footprint.")
```

```python
# Structured output instead of a prose report
analyzer = RepoRLM(
    "/repos/requests",
    "task -> public_api: list[str], dependencies: list[str], summary: str",
)
r = analyzer(task="Audit for maintenance risk.")
print(r.public_api, r.dependencies, r.summary)
```

Repos mount under `/repos/<name>`. Pass `mode="overlay"` to let the model write
scratch files (captured in memory; the host is never modified). The default
signature is `task -> report: str`.

## Why Monty?

- **Fast**: Microsecond startup (no subprocess, no WASM bootstrap)
- **Secure**: No filesystem, network, or environment access by default
- **Lightweight**: Pure Rust, no Deno/Pyodide dependency
