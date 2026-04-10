# dspy-monty-interpreter

DSPy `CodeInterpreter` implementation using [Monty](https://github.com/pydantic/monty), a secure Python interpreter written in Rust.

The Monty team points out, "This project is still in development, and not ready for the prime time." It uses a small subset of the standard library (`sys`, `os`, `typing`, `asyncio`, `re`, `datetime`, `json`) and can't yet define classes or use match statements. 

That said: Monty is *fast*. For many RLM use cases, Monty is my daily driver.

## Installation

```bash
pip install dspy-monty-interpreter
```

Requires `pydantic-monty>=0.0.10`.

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

## Why Monty?

- **Fast**: Microsecond startup (no subprocess, no WASM bootstrap)
- **Secure**: No filesystem, network, or environment access by default
- **Lightweight**: Pure Rust, no Deno/Pyodide dependency
