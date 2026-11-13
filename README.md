# dspy-monty-interpreter

DSPy `CodeInterpreter` implementation using [Monty](https://github.com/pydantic/monty), a secure Python interpreter written in Rust.

Monty describes itself as "a sandbox for Python code written by AI." It implements a subset of the standard library (`sys`, `os`, `pathlib`, `typing`, `asyncio`, `re`, `datetime`, `time`, `random`, `json`, `math`, `unicodedata`, `collections`, `itertools`, `functools`, `dataclasses`, `copy`, `base64`, `binascii`, `csv`, `tomllib`, `hashlib`, `bisect`, `decimal`, `fnmatch`, `annotationlib`, `os.path`) and can't yet use generator functions (`yield`) or class inheritance. The adapter makes subclasses of `dspy.Module` and `dspy.Signature` work (see [dspy.Flex](#dspyflex)). It does support classes, `match` statements, decorators, `complex` numbers, `@dataclass` (including `frozen=` and `eq=`), generator expressions, `with`/context managers, and a sandboxed `open()` (file access is opt-in, see [Filesystem access](#filesystem-access)).

That said: Monty is *fast*. For many RLM use cases, Monty is my daily driver.

## Installation

```bash
pip install dspy-monty-interpreter
```

## Usage

```python
import dspy
from dspy_monty_interpreter import MontyInterpreter

rlm = dspy.RLM("context -> answer", interpreter_factory=MontyInterpreter)
result = rlm(context="What is 2 + 2?")
```

DSPy calls `interpreter_factory` once per `forward()` and shuts the interpreter down afterwards. Starting a Monty worker pool takes a few milliseconds, so this is the right default even under `dspy.Evaluate`. To configure the interpreter, use `MontyInterpreter.factory(...)`, which takes the same arguments as the constructor:

```python
rlm = dspy.RLM(
    "context -> answer",
    interpreter_factory=MontyInterpreter.factory(request_timeout=10.0),
)
```

Since DSPy 3.4 you can also make Monty the default runtime for every code-executing module with `dspy.configure` (or scope it with `dspy.context`), and override it for a single call:

```python
dspy.configure(interpreter_factory=MontyInterpreter.factory(request_timeout=10.0))

rlm = dspy.RLM("context -> answer")  # runs on Monty
result = rlm(context="What is 2 + 2?")

# One call with a different configuration:
result = rlm(
    context="Which report mentions Q3?",
    interpreter_factory=MontyInterpreter.factory(mounts=[...]),
)
```

RLM reads `execution_instructions` off whichever factory is active for each call, so the prompt always describes the runtime that will run the code, mounted directories included.

DSPy 3.4 removed the caller-owned interpreter (`rlm(interpreter, ...)`): every interpreter now belongs to the `forward()` that created it. A `MontyInterpreter` instance is still useful on its own, for example in tests or benchmarks, through `execute()`.

Tools may be `async def` functions. RLM wraps them, and `MontyInterpreter` awaits the result when sandbox code calls them, both from `forward()` and from `acall()`.

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
Mounted directories (explore with os.listdir(path), pathlib.Path(path).iterdir(), os.walk(path), and open(path)):
- /data (read-only) containing q3/, readme.md
- /scratch (overlay: writable, but writes are discarded when each execution ends) containing (empty)
```

The sandbox's working directory starts at the first mount's virtual path (`/data` above, or `/` without mounts), so relative paths like `open('q3/summary.md')` resolve inside it, and `os.chdir()` persists across `execute()` calls.

Sandboxed code can also `import` `.py` files from a mounted directory: with `helpers.py` in `./reports`, the example above lets the model run `import helpers`. Imported files run inside the sandbox under the same limits as any other code. Mount only code you are willing to let the model run.

The listing is a snapshot of each host directory's top level at the time the factory is created, and it is bounded so a large mount cannot bloat the prompt. Up to 20 entries (`mount_listing_limit`) are listed by name; beyond that the model gets a summary instead, for example `1,204 files and 1 directory (.csv ×1200, .md ×4; e.g. 0000.csv, 0001.csv, 0002.csv)`. Individual names are cut at 40 characters, each mount's description at roughly 300, and the directory scan stops at 5,000 entries. Pass `mount_listing_limit=0` to describe only the path and mode. Without mounts the section says the filesystem is unavailable, so the model does not go looking for one. An instance created by the factory carries the same `execution_instructions` if you need them elsewhere.

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

## Host modules

Pass `external_modules` to let sandbox code `import` modules you build on the host. Each entry maps a module name to a `pydantic_monty.ExternalModule`, or to a zero-argument function that returns one (called once per session, the first time the code imports it):

```python
import dspy
from dspy_monty_interpreter import ClassInstance, ExternalModule, MontyInterpreter


class Classifier:
    def __init__(self, labels: list[str]):
        self.labels = labels
        self._predict = dspy.Predict("text, labels: list[str] -> label: str")

    def classify(self, text: str) -> str:
        return self._predict(text=text, labels=self.labels).label


def classifier(labels: list[str]):
    return ClassInstance(Classifier(labels), eager_attrs={"labels"}, allowed_methods={"classify"})


def summarize(text: str) -> str:
    return dspy.Predict("text -> summary")(text=text).summary


helpers = ExternalModule({"summarize": summarize, "classifier": classifier})
rlm = dspy.RLM(
    "documents -> report",
    interpreter_factory=MontyInterpreter.factory(external_modules={"helpers": helpers}),
)
```

The sandbox code can then run `from helpers import summarize` or `c = helpers.classifier(["bug", "feature"])` followed by `c.classify(text)`. A few rules:

- Functions in a module run on the host, outside the sandbox. They can return plain data (strings, numbers, lists, dicts, dataclasses). Wrap any other object in `ClassInstance` and list the methods the sandbox may call. A wrapped object can't be called directly, so expose a method such as `classify()` in place of `__call__`.
- Monty can't take a Python module as is: `external_modules={"dspy": dspy}` fails. Build an `ExternalModule` with exactly the functions you want to expose.
- Keys starting with `_` are hidden from the sandbox. Nested `ExternalModule`s become submodules (`import helpers.text`).
- Calls into a module don't fire DSPy's tool callbacks. Use `tools` when you need them traced.

The sandbox has no `dir()` or `help()`, so the model learns a module's API from the prompt. `execution_instructions` lists each module's public attributes and function signatures, or shows its stubs when you pass `ExternalModule(attrs, stubs="def summarize(text: str) -> str: ...")`. Stubs over 4,000 characters are truncated. A module given as a factory is listed by name only, so the factory isn't called before the code needs it.

## Object inputs

RLM inputs that aren't plain data, such as your own classes or pydantic models, reach the sandbox as read-only proxies: the code can read their public attributes but can't call their methods. To expose a method, wrap the object in `ClassInstance` and list it in `allowed_methods`.

## Sub-agents

On DSPy 3.5 and later, RLM code can `import dspy` and build sub-agents with `dspy.Predict`, `dspy.ChainOfThought`, `dspy.ReAct`, and `dspy.RLM`. The sub-agents run on the host with your LM. Their LM calls count against the RLM's `max_llm_calls`. Their results support `r.answer`, `r["answer"]`, and `"answer" in r`.

A `ReAct` sub-agent accepts the tools you passed to the RLM and functions the model defines in the REPL. A REPL function runs inside the sandbox and can read the REPL's variables. The agent itself runs on the host. The adapter builds the tool description from the function's signature and docstring. A nested `RLM` sub-agent accepts only the tools you passed.

## dspy.Flex

`dspy.Flex` runs a `dspy.Module` whose source code `dspy.GEPA` can rewrite. Pass the factory as its interpreter, or make Monty the default with `dspy.configure`:

```python
flex = dspy.Flex("question -> answer", interpreter_factory=MontyInterpreter)
result = flex(question="What is 2 + 2?")
```

Monty has no class inheritance. The adapter therefore rewrites the module source before it runs. A `dspy.Module` subclass loses its base and its `super().__init__()` call and gains a `__call__` that runs `forward`. A `dspy.Signature` class becomes a signature the host builds from its field declarations. The rewrite keeps line numbers. Error messages therefore point at the code as written. Module code can call Flex tools by name. Values of the signature's custom types arrive as plain dicts. Every predictor call counts against `max_predictor_calls`.

GEPA's code proposer doesn't see the interpreter's limits. A proposed module may use a feature Monty lacks, such as `yield`. That candidate fails with an error, and GEPA's reflection reads the error.

## Type checking

Pass `type_check=True` to check each code block with Monty's type checker before it runs:

```python
rlm = dspy.RLM(
    "context -> answer",
    tools=[search],
    interpreter_factory=MontyInterpreter.factory(type_check=True),
)
```

The checker knows your tools' signatures, the input variables' types, and the sandbox's `dspy` namespace. It catches a wrong argument type, a misspelled name, or `context + 1` on a string input before any of the block runs. A rejected block leaves the REPL state untouched. The prompt tells the model about the check. Variables the code assigns may change type in later blocks. The checker is strict. It rejects `d.get(key) + 1`, for example, because `get` may return `None`. It suits models that write careful code.

## Conversations

Each RLM call gets a fresh interpreter. REPL variables therefore vanish between the turns of a conversation. Pass a `SessionState` to keep them:

```python
from dspy_monty_interpreter import SessionState

state = SessionState()  # one per conversation
rlm = dspy.RLM("question -> answer", interpreter_factory=MontyInterpreter.factory(state=state))

first = rlm(question="Load the sales figures into a variable.")
second = rlm(question="Now chart them by month.", history=first.history)
```

The interpreter saves its session into `state` when it shuts down, and the next interpreter restores it. Variables, functions, and classes carry over. Input proxies and sub-agents from an earlier turn don't carry over and must be made again. `state.data` holds the serialized session as bytes. You can store those bytes between processes. Load only bytes your own code produced.

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

Two keys cap execution time: `max_feed_duration_secs` limits one `execute()` call, and `max_turn_duration_secs` limits the sandbox time between tool calls. Both count only time spent running sandboxed code, not time waiting on your tools. `max_suspensions` caps the host calls (tools, `llm_query`, sub-agents) one session may make. Monty's default of 1000 is too low for an RLM that loops a tool over its data. The adapter raises it to 1,000,000. Other keys are `max_recursion_depth` (default 1000), `max_total_sleep_secs`, and `gc_interval`. Monty rejects unknown keys with a `ValueError` on the first `execute()`.

A limit violation surfaces as a `CodeExecutionError` from `execute()`, which RLM feeds back to the model. After a memory, recursion, or suspension limit the session keeps its state. After a time limit Monty makes no guarantees about the sandbox heap, so the adapter discards the session: the next `execute()` starts with empty state, and the error message tells the model so.

## Timeouts

Pass `request_timeout` to set a hard limit, in seconds, on each `execute()` call:

```python
interp = MontyInterpreter(request_timeout=10.0)
```

When code exceeds the limit, `execute()` raises `CodeInterpreterError`. The session state is lost, and the next `execute()` starts fresh. (Ordinary runtime errors in the code raise `CodeExecutionError`, a subclass that RLM feeds back to the model as a correction turn; a timeout is terminal and ends the RLM run.)

## Remote sandboxes

Pass `remote` with the URL of a [`monty-server`](https://github.com/pydantic/monty) to run code on a remote machine:

```python
interp_factory = MontyInterpreter.factory(remote="https://sandbox.internal:8799", request_timeout=30.0)
```

Tools, `llm_query`, and sub-agents still run in your process, and the remote worker calls back to them. `request_timeout` and `max_processes` configure the remote pool. For other options, such as authentication metadata, pass a zero-argument callable that returns a `pydantic_monty.AsyncMontyRemote`. A dropped or refused connection raises `CodeInterpreterError`. RLM ends the run on that error.

## Parallel evaluation

`dspy.Evaluate` and `dspy.Parallel` call `forward()` concurrently from several threads. Each `forward()` creates its own `MontyInterpreter` from the factory, with its own worker pool, and shuts it down when it finishes, so the threads share nothing and there is nothing to configure:

```python
rlm = dspy.RLM(
    "question -> answer",
    interpreter_factory=MontyInterpreter.factory(request_timeout=30.0),
)
evaluate = dspy.Evaluate(devset=devset, num_threads=8, metric=metric)
evaluate(rlm)
```

A pool starts its worker lazily on the first `execute()`, in a few milliseconds, so per-forward pools cost little. `max_processes` only matters when you drive one interpreter instance from several threads yourself, outside RLM: sessions are per-thread and share that instance's pool, and a thread holds its worker between `execute()` calls, so set `max_processes` to at least the thread count.

## Observability

`MontyInterpreter` fires DSPy's interpreter lifecycle callbacks: `on_interpreter_startup_*`, `on_interpreter_execute_*`, `on_interpreter_tool_call_*`, and `on_interpreter_shutdown_*`. Register a `dspy.utils.callback.BaseCallback` globally with `dspy.configure(callbacks=[...])` or per instance with `MontyInterpreter(callbacks=[...])`. Tool calls nest under the `execute()` call that made them, so traces keep their ancestry.

With `opentelemetry-api` installed (`pip install dspy-monty-interpreter[telemetry]`), each `execute()` call also records a `monty.execute` span with its outcome: `output`, `none`, `submit`, or `error`. Call `instrument()` once to add Monty's own spans and metrics. Monty records a span per session, with a child span for each code run and a grandchild for each host call. The session span starts inside the first `monty.execute` of the session:

```python
from dspy_monty_interpreter import instrument

instrument()  # uses the global OpenTelemetry providers
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

- **Fast**: No WASM bootstrap. Code runs against a pool of warm `monty` worker processes
- **Secure**: No filesystem, network, or environment access unless you grant it (see [Filesystem access](#filesystem-access))
- **Resilient**: A crashed or timed-out worker is replaced automatically without taking down your process
- **Lightweight**: Pure Rust, no Deno/Pyodide dependency
