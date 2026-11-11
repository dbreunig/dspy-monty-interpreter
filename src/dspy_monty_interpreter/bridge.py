"""Bridge between dspy's sandbox session (dspy 3.5+) and Monty.

dspy 3.5 makes sub-agents and non-JSON inputs available in an RLM's REPL,
and runs ``dspy.Flex`` modules, by running a bootstrap in the interpreter.
Monty cannot run that bootstrap, so the adapter answers it on the host. dspy
then treats the interpreter as one that shares the host's memory: it
rebuilds inputs on the host and runs sub-agents on the host's dspy.

Sandbox code sees ``dspy`` as a small namespace defined in Monty by
``PRELUDE``. Its ``Prediction``, ``Module`` and ``Tool`` are sandbox
classes, so predictions support ``p["answer"]`` and ``"answer" in p`` and
modules call their ``forward``. Its sub-agents (``Predict``,
``ChainOfThought``, ``ReAct``, ``RLM``) and ``Signature`` are built and run
on the host through the ``_dspy_host`` module. ``dspy.ReAct`` also accepts
functions defined in the REPL: the agent runs on a host thread, and each
call it makes to a REPL function pauses it and returns to the sandbox, which
runs the function and resumes the agent with the result.

Host objects cross with the narrowest exposure that works. An input that is
not plain data exposes only its public data attributes; a caller who wants
the sandbox to call methods wraps the object in ``ClassInstance`` with an
explicit ``allowed_methods``. A sub-agent is an opaque handle, so sandbox
code cannot reach module methods such as ``save()`` or ``load()``.
"""

from __future__ import annotations

import contextvars
import datetime
import decimal
import inspect
import io
import pathlib
import queue
import re
import threading
import tokenize
from collections.abc import Callable, Mapping
from typing import Any

from dspy.primitives.prediction import Prediction
from pydantic_monty import ClassInstance, ClassType, ExternalModule

from dspy_monty_interpreter.rewrite import FunctionInfo

try:
    from dspy.primitives import facade as _facade
except ImportError:  # dspy < 3.5 has no sandbox session
    _facade = None

SUPPORTED = (
    _facade is not None
    and isinstance(getattr(_facade, "_in_host", None), dict)
    and "def __dspy_bootstrap" in getattr(_facade, "BOOTSTRAP", "")
)

# dspy names the variable that carries a rebuilt input ``_raw_<name>``.
RAW_PREFIX = "_raw_"
# dspy registers its bridge tools under this prefix.
_BRIDGE_TOOL_PREFIX = "__dspy"
# The host tool dspy routes provided-tool calls through (sandbox_guest.CALL_TOOL).
_CALL_TOOL = "__dspy_tool_call__"
# The external module the prelude imports; sandbox code uses ``dspy`` instead.
HOST_MODULE = "_dspy_host"

PROXY_INSTRUCTIONS = (
    "Inputs that are not plain data are proxies for objects outside the sandbox: "
    "repr(x) shows their data attributes, which you can read."
)
SUB_AGENT_INSTRUCTIONS = (
    "In the sandbox, `dspy` provides Predict, ChainOfThought, ReAct, RLM, Signature, Prediction, Module, and Tool. "
    "Sub-agents run outside the sandbox. ReAct accepts the provided tools and functions you define in the REPL; "
    "RLM accepts only the provided tools."
)

# The prelude's names as the type checker sees them. Predictions declare
# __getattr__ so any field reads as Any; Monty itself sets the fields.
PRELUDE_STUBS = """\
class _DspyPrediction:
    def __init__(self, **fields: Any) -> None: ...
    def __getattr__(self, name: str) -> Any: ...
    def __getitem__(self, name: str) -> Any: ...
    def __contains__(self, name: str) -> bool: ...
    def __iter__(self) -> Any: ...
    def get(self, name: str, default: Any = None) -> Any: ...
    def keys(self) -> list[str]: ...
    def values(self) -> list[Any]: ...
    def items(self) -> list[tuple[str, Any]]: ...
    def toDict(self) -> dict[str, Any]: ...
class _DspyModule:
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
class _DspyTool:
    def __init__(self, func: Any, name: str | None = None, desc: str | None = None) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
class _DspyAgent:
    def __call__(self, **kwargs: Any) -> _DspyPrediction: ...
class _DspyNamespace:
    Prediction: type[_DspyPrediction]
    Module: type[_DspyModule]
    Tool: type[_DspyTool]
    def Signature(self, signature: str, instructions: str | None = None, fields: Any = None,
                  custom_types: Any = None) -> Any: ...
    def InputField(self, **kwargs: Any) -> Any: ...
    def OutputField(self, **kwargs: Any) -> Any: ...
    def Predict(self, signature: Any, **kwargs: Any) -> _DspyAgent: ...
    def ChainOfThought(self, signature: Any, **kwargs: Any) -> _DspyAgent: ...
    def ReAct(self, signature: Any, tools: list[Any] | None = None, **kwargs: Any) -> _DspyAgent: ...
    def RLM(self, signature: Any, tools: list[Any] | None = None, **kwargs: Any) -> _DspyAgent: ...
dspy: _DspyNamespace
"""

PRELUDE = '''
import _dspy_host


class _DspyPrediction:
    def __init__(self, **fields):
        self._store = dict(fields)
        for name in fields:
            # A field named like a method stays reachable as p[name].
            if name not in ("_store", "get", "keys", "values", "items", "toDict"):
                setattr(self, name, fields[name])

    def __getitem__(self, name):
        return self._store[name]

    def __contains__(self, name):
        return name in self._store

    def __iter__(self):
        return iter(self._store)

    def __repr__(self):
        return "Prediction(" + ", ".join(k + "=" + repr(v) for k, v in self._store.items()) + ")"

    def get(self, name, default=None):
        return self._store.get(name, default)

    def keys(self):
        return list(self._store.keys())

    def values(self):
        return list(self._store.values())

    def items(self):
        return list(self._store.items())

    def toDict(self):
        return dict(self._store)


class _DspyModule:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class _DspyTool:
    def __init__(self, func, name=None, desc=None):
        self.func = func
        self.name = name
        self.desc = desc

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class _DspyAgent:
    def __init__(self, handle, tools):
        self._handle = handle
        self._tools = tools

    def __call__(self, **kwargs):
        reply = _dspy_host.call(self._handle, kwargs)
        while reply["kind"] == "call":
            try:
                answer = {"value": self._tools[reply["index"]](**reply["kwargs"])}
            except Exception as e:
                answer = {"error": type(e).__name__ + ": " + str(e)}
            reply = _dspy_host.resume(self._handle, answer)
        return _DspyPrediction(**reply["fields"])


def _dspy_tools(tools):
    funcs = []
    specs = []
    for tool in tools or []:
        if isinstance(tool, _DspyTool):
            funcs.append(tool.func)
            specs.append({"ref": tool.func, "name": tool.name, "desc": tool.desc})
        else:
            funcs.append(tool)
            specs.append({"ref": tool})
    return funcs, specs


class _DspyNamespace:
    Prediction = _DspyPrediction
    Module = _DspyModule
    Tool = _DspyTool

    def Signature(self, signature, instructions=None, fields=None, custom_types=None):
        return _dspy_host.signature(signature, instructions, fields, custom_types)

    def InputField(self, **kwargs):
        return kwargs

    def OutputField(self, **kwargs):
        return kwargs

    def Predict(self, signature, **kwargs):
        return _DspyAgent(_dspy_host.build("Predict", signature, kwargs, []), [])

    def ChainOfThought(self, signature, **kwargs):
        return _DspyAgent(_dspy_host.build("ChainOfThought", signature, kwargs, []), [])

    def ReAct(self, signature, tools=None, **kwargs):
        funcs, specs = _dspy_tools(tools)
        return _DspyAgent(_dspy_host.build("ReAct", signature, kwargs, specs), funcs)

    def RLM(self, signature, tools=None, **kwargs):
        funcs, specs = _dspy_tools(tools)
        return _DspyAgent(_dspy_host.build("RLM", signature, kwargs, specs), funcs)


dspy = _DspyNamespace()
'''

# Scalars Monty converts natively.
_PLAIN_TYPES = (
    type(None),
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    datetime.date,
    datetime.time,
    datetime.timedelta,
    decimal.Decimal,
    pathlib.PurePath,
)
# Values that cross unchanged; everything else crosses as a proxy.
_NATIVE_TYPES = (
    *_PLAIN_TYPES,
    ClassInstance,
    ClassType,
    ExternalModule,
)
# How a REPL function reaches the host when sandbox code passes it as a value.
_REPL_FUNCTION_RE = re.compile(r"<function (?:[\w.<>]+\.)?(\w+) at 0x[0-9a-f]+>")
_SIMPLE_ANNOTATIONS = {"str": str, "int": int, "float": float, "bool": bool, "list": list, "dict": dict}


def instructions() -> str:
    """The bridge's part of the interpreter's execution instructions."""
    if SUPPORTED:
        return f"{PROXY_INSTRUCTIONS} {SUB_AGENT_INSTRUCTIONS}"
    return PROXY_INSTRUCTIONS


def is_bridge_tool(name: str) -> bool:
    """Whether ``name`` is one of dspy's own bridge tools. They serve
    sandboxes that run dspy's guest; in Monty, sub-agents run on the host."""
    return name.startswith(_BRIDGE_TOOL_PREFIX)


def is_bootstrap(code: str, variables: Mapping[str, Any] | None) -> bool:
    """Whether ``execute(code, variables)`` is dspy installing its sandbox session."""
    return SUPPORTED and bool(variables) and "__dspy_spec" in variables and code == _facade.BOOTSTRAP


def answer_bootstrap(variables: Mapping[str, Any]) -> dict[str, Any]:
    """Register a host namespace for the session, as the bootstrap does in a
    process that shares the host's memory. dspy fills it with the session's
    spec, provided tools and custom types."""
    namespace: dict[str, Any] = {}
    _facade._in_host[variables["__dspy_spec"]["session"]] = namespace
    return namespace


def to_monty(value: Any) -> Any:
    """``value`` as sandbox code should see it: Monty-native values unchanged,
    containers converted item by item, other objects as data-only proxies."""
    if isinstance(value, _NATIVE_TYPES) or inspect.isroutine(value):
        return value
    if isinstance(value, list):
        return [to_monty(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_monty(item) for item in value)
    if isinstance(value, dict):
        return {key: to_monty(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return value
    return HostObject.wrap(value)


def to_plain(value: Any) -> Any:
    """``value`` with pydantic models as the JSON data they write, as
    ``dspy.Flex`` promises its modules; other values as ``to_monty``."""
    from pydantic import BaseModel

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [to_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: to_plain(item) for key, item in value.items()}
    return to_monty(value)


def _is_plain(value: Any) -> bool:
    """Whether ``value`` crosses without a proxy or a host callable."""
    if isinstance(value, (list, tuple, set, frozenset)):
        return all(_is_plain(item) for item in value)
    if isinstance(value, dict):
        return all(_is_plain(item) for item in value.values())
    return isinstance(value, _PLAIN_TYPES)


class HostObject(ClassInstance):
    """A host object in the sandbox, exposed as data only: public data
    attributes are copied in, other public attributes are fetched on demand,
    and no method runs. Callables and classes never cross."""

    @classmethod
    def wrap(cls, value: Any) -> HostObject:
        return cls(value, eager_attrs=_plain_public_attrs(value), lazy_attrs="all")

    def lookup_lazy_attrs(self, name: str) -> Any:
        if name.startswith("_"):
            raise self.attr_error(name)
        return super().lookup_lazy_attrs(name)

    def convert_value(self, /, name: str, value: Any) -> Any:
        if callable(value):
            raise self.attr_error(name)
        return to_monty(value)


def _plain_public_attrs(value: Any) -> list[str]:
    if isinstance(value, Prediction):
        items = value.items()
    else:
        items = getattr(value, "__dict__", {}).items()
    return [name for name, item in items if not name.startswith("_") and _is_plain(item)]


class BridgeContext:
    """What the ``_dspy_host`` module needs from its interpreter: the live
    tool mapping, dspy's session namespace once installed, and the functions
    REPL code has defined."""

    def __init__(self, tools: Mapping[str, Callable[..., Any]]) -> None:
        self.tools = tools
        self.namespace: dict[str, Any] | None = None
        self.functions: dict[str, FunctionInfo] = {}
        self._runs: set[_AgentRun] = set()
        self._session_tools: dict[str, Any] = {}
        self._lock = threading.Lock()

    def custom_types(self) -> dict[str, type]:
        """The custom types of the session's signature (a Flex's), by name."""
        return dict((self.namespace or {}).get("__dspy_types") or {})

    def session_tools(self) -> dict[str, dict[str, Any]]:
        """Provided tools that dspy serves through its tool-call host tool
        (a Flex's), by name, as dspy describes them."""
        spec = (self.namespace or {}).get("__dspy_guest_spec") or {}
        if _CALL_TOOL not in self.tools:
            return {}
        return {name: tool for name, tool in (spec.get("tools") or {}).items() if name not in self.tools}

    def session_tool(self, name: str) -> Any:
        """The ``dspy.Tool`` for a session tool; its ``func`` binds positional
        arguments in the order dspy declares them."""
        with self._lock:
            tool = self._session_tools.get(name)
        if tool is None:
            tool = _session_tool(self, name, self.session_tools()[name])
            with self._lock:
                tool = self._session_tools.setdefault(name, tool)
        return tool

    def call_session_tool(self, name: str, kwargs: dict[str, Any]) -> Any:
        return self.tools[_CALL_TOOL](tool=name, arguments=kwargs)

    def provided_tools(self) -> dict[str, Callable[..., Any]]:
        """Every provided tool sandbox code may hand to a sub-agent, by name."""
        tools = {name: fn for name, fn in self.tools.items() if not is_bridge_tool(name)}
        for name in self.session_tools():
            tools[name] = self.session_tool(name)
        return tools

    def track(self, run: _AgentRun) -> None:
        with self._lock:
            self._runs.add(run)

    def release(self, run: _AgentRun) -> None:
        with self._lock:
            self._runs.discard(run)

    def cancel_runs(self, owner: int | None = None) -> None:
        """Stop the agent runs still paused at a REPL-function call: those
        started by thread ``owner``, or all of them. Sandbox code resumes a
        run only within the snippet that started it, so a run still paused
        when its snippet ends is orphaned."""
        with self._lock:
            runs = [r for r in self._runs if owner is None or r.owner == owner]
            self._runs.difference_update(runs)
        for run in runs:
            run.cancel()


def _session_tool(ctx: BridgeContext, name: str, spec: dict[str, Any]) -> Any:
    import dspy

    args = dict(spec.get("args") or {})

    def call(*positional: Any, **kwargs: Any) -> Any:
        if len(positional) > len(args):
            raise TypeError(f"{name}() takes {len(args)} positional arguments but {len(positional)} were given")
        kwargs.update(zip(args, positional, strict=False))
        return ctx.call_session_tool(name, kwargs)

    # Callers such as RLM bind positional arguments against the signature.
    params = []
    kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
    for arg, schema in args.items():
        default = (schema if isinstance(schema, dict) else {}).get("default", inspect.Parameter.empty)
        if default is inspect.Parameter.empty and params and params[-1].default is not inspect.Parameter.empty:
            kind = inspect.Parameter.KEYWORD_ONLY
        params.append(inspect.Parameter(arg, kind, default=default))
    call.__name__ = name
    call.__signature__ = inspect.Signature(params)
    return dspy.Tool(call, name=name, desc=spec.get("desc") or "", args=args)


class _AgentCancelled(BaseException):
    """Ends a cancelled agent run. A BaseException, so ReAct's own error
    handling, which catches Exception, lets it through."""


# The agent run a REPL-function stub belongs to, set on the run's thread.
_current_run: contextvars.ContextVar[_AgentRun | None] = contextvars.ContextVar("dspy_monty_agent_run", default=None)


class _AgentRun:
    """One call of a threaded agent: its thread, and the queues it shares
    with the sandbox loop that drives it."""

    def __init__(self, ctx: BridgeContext) -> None:
        self.owner = threading.get_ident()  # the thread whose sandbox drives this run
        self.events: queue.Queue = queue.Queue()
        self.replies: queue.Queue = queue.Queue()
        self.cancelled = threading.Event()
        self._ctx = ctx

    def start(self, module: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        self._ctx.track(self)
        context = contextvars.copy_context()
        threading.Thread(target=context.run, args=(self._run, module, kwargs), daemon=True).start()
        return self.next()

    def resume(self, answer: dict[str, Any]) -> dict[str, Any]:
        self.replies.put(answer)
        return self.next()

    def ask_sandbox(self, index: int, kwargs: dict[str, Any]) -> Any:
        """Run on the agent's thread: hand a REPL-function call to the sandbox."""
        if self.cancelled.is_set():
            raise _AgentCancelled
        self.events.put(("call", index, kwargs))
        reply = self.replies.get()
        if reply is None:
            raise _AgentCancelled
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply.get("value")

    def cancel(self) -> None:
        self.cancelled.set()
        self.replies.put(None)

    def _run(self, module: Any, kwargs: dict[str, Any]) -> None:
        _current_run.set(self)
        try:
            self.events.put(("done", module(**kwargs)))
        except _AgentCancelled:
            pass
        except BaseException as e:
            self.events.put(("error", e))

    def next(self) -> dict[str, Any]:
        event = self.events.get()
        if event[0] == "call":
            return {"kind": "call", "index": event[1], "kwargs": to_monty(event[2])}
        self._ctx.release(self)
        if event[0] == "error":
            raise event[1]
        return {"kind": "done", "fields": _fields(event[1])}


class HostAgent:
    """A dspy module built for sandbox code. With REPL-function tools each
    call runs on its own thread, which pauses at each such call until the
    sandbox answers it through ``resume``."""

    def __init__(self, ctx: BridgeContext, module: Any, threaded: bool) -> None:
        self.module = module
        self._ctx = ctx
        self._threaded = threaded
        self._run: _AgentRun | None = None

    def call(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        if not self._threaded:
            return {"kind": "done", "fields": _fields(self.module(**kwargs))}
        if self._run is not None:
            self._run.cancel()  # left paused by a snippet that ended early
        self._run = _AgentRun(self._ctx)
        return self._finish(self._run.start(self.module, kwargs))

    def resume(self, answer: dict[str, Any]) -> dict[str, Any]:
        if self._run is None or self._run.cancelled.is_set():
            raise RuntimeError("This sub-agent has no call waiting for a result.")
        return self._finish(self._run.resume(answer))

    def _finish(self, reply: dict[str, Any]) -> dict[str, Any]:
        if reply["kind"] == "done":
            self._run = None
        return reply


def _ask_sandbox(index: int, kwargs: dict[str, Any]) -> Any:
    run = _current_run.get()
    if run is None:
        raise RuntimeError("A REPL function can only run while its sub-agent is running.")
    return run.ask_sandbox(index, kwargs)


def _fields(prediction: Any) -> dict[str, Any]:
    items = prediction.items() if isinstance(prediction, Prediction) else dict(prediction).items()
    return {name: to_monty(value) for name, value in items}


def dspy_module(ctx: BridgeContext) -> ExternalModule:
    """The host side of the sandbox's ``dspy`` namespace."""
    import dspy

    kinds = {"Predict": dspy.Predict, "ChainOfThought": dspy.ChainOfThought, "ReAct": dspy.ReAct, "RLM": dspy.RLM}

    def build(kind: str, signature: Any, kwargs: dict[str, Any], specs: list[dict[str, Any]]) -> ClassInstance:
        if kind not in kinds:
            raise TypeError(f"Unknown sub-agent kind {kind!r}")
        signature = _signature_arg(ctx, signature)
        kwargs = dict(kwargs)
        if kind not in ("ReAct", "RLM"):
            agent = HostAgent(ctx, kinds[kind](signature, **kwargs), threaded=False)
            return ClassInstance(agent)
        provided = ctx.provided_tools()
        tools = [_resolve_tool(ctx, kind, i, spec, provided) for i, spec in enumerate(specs)]
        threaded = any(getattr(getattr(t, "func", None), "__dspy_repl_function__", False) for t in tools)
        return ClassInstance(HostAgent(ctx, kinds[kind](signature, tools=tools, **kwargs), threaded=threaded))

    def call(handle: HostAgent, kwargs: dict[str, Any]) -> dict[str, Any]:
        return _agent(handle).call(kwargs)

    def resume(handle: HostAgent, answer: dict[str, Any]) -> dict[str, Any]:
        return _agent(handle).resume(answer)

    def signature(
        signature: Any,
        instructions: str | None = None,
        fields: dict[str, Any] | None = None,
        custom_types: dict[str, Any] | None = None,
    ) -> ClassInstance:
        types = {**ctx.custom_types(), **_resolve_types(ctx, custom_types)}
        if not isinstance(signature, str):
            raise TypeError("dspy.Signature() takes a signature string, like 'question -> answer'.")
        spec = {"signature": signature, "instructions": instructions, "fields": fields or {}}
        return ClassInstance(_facade._resolve_signature(spec, types), eager_attrs=("instructions",))

    return ExternalModule({"build": build, "call": call, "resume": resume, "signature": signature})


def _agent(handle: Any) -> HostAgent:
    if not isinstance(handle, HostAgent):
        raise TypeError("Not a sub-agent handle.")
    return handle


def _signature_arg(ctx: BridgeContext, signature: Any) -> Any:
    """A sub-agent's signature: a string resolved against the session's custom
    types, or a signature class built by ``dspy.Signature``."""
    if isinstance(signature, str):
        types = ctx.custom_types()
        if types:
            from dspy.signatures.signature import make_signature

            return make_signature(signature, custom_types=types)
        return signature
    from dspy.signatures.signature import Signature

    if isinstance(signature, type) and issubclass(signature, Signature):
        return signature
    raise TypeError("A sub-agent's signature must be a string or a dspy.Signature.")


def _resolve_types(ctx: BridgeContext, custom_types: dict[str, Any] | None) -> dict[str, type]:
    """Custom types sandbox code names; each reaches the host as its name."""
    known = ctx.custom_types()
    resolved = {}
    for key, value in (custom_types or {}).items():
        name = value if isinstance(value, str) else key
        if name not in known:
            raise TypeError(f"Unknown custom type {name!r}; the signature's types are {sorted(known) or 'none'}.")
        resolved[key] = known[name]
    return resolved


def _resolve_tool(
    ctx: BridgeContext, kind: str, index: int, spec: dict[str, Any], provided: dict[str, Callable[..., Any]]
) -> Any:
    """A sub-agent's tool from what sandbox code passed: a provided tool,
    which reaches the host as its name, or (for ReAct) a REPL function,
    which reaches it as its repr and runs back in the sandbox."""
    import dspy

    ref = spec.get("ref")
    if isinstance(ref, str) and ref in provided:
        tool = provided[ref]
        if spec.get("name") or spec.get("desc"):
            tool = dspy.Tool(tool.func if isinstance(tool, dspy.Tool) else tool, name=spec.get("name"),
                             desc=spec.get("desc"))
        return tool
    m = _REPL_FUNCTION_RE.fullmatch(ref) if isinstance(ref, str) else None
    if m and kind == "ReAct":
        info = ctx.functions.get(m[1], FunctionInfo(m[1]))
        stub = _repl_function_stub(info, index)
        return dspy.Tool(stub, name=spec.get("name") or info.name, desc=spec.get("desc") or info.doc or None)
    names = ", ".join(sorted(provided)) or "none"
    accepted = "the provided tools or functions defined in the REPL" if kind == "ReAct" else "the provided tools"
    raise TypeError(f"{kind} tools must be {accepted} (provided: {names}); got {ref!r}")


def _repl_function_stub(info: FunctionInfo, index: int) -> Callable[..., Any]:
    """A host function shaped like a REPL function, for the tool-calling
    prompt, that hands each call back to the sandbox."""

    def stub(**kwargs: Any) -> Any:
        return _ask_sandbox(index, kwargs)

    stub.__dspy_repl_function__ = True

    params = []
    annotations: dict[str, Any] = {}
    for p in info.params:
        default = p.get("default", inspect.Parameter.empty)
        annotation = _SIMPLE_ANNOTATIONS.get(p.get("annotation", ""), inspect.Parameter.empty)
        params.append(inspect.Parameter(p["name"], inspect.Parameter.KEYWORD_ONLY, default=default,
                                        annotation=annotation))
        if annotation is not inspect.Parameter.empty:
            annotations[p["name"]] = annotation
    stub.__name__ = stub.__qualname__ = info.name
    stub.__doc__ = info.doc or None
    stub.__signature__ = inspect.Signature(params)
    stub.__annotations__ = annotations
    return stub


def take_raw_variables(code: str, variables: dict[str, Any]) -> tuple[str, dict[str, Callable[[], Any]]]:
    """Move dspy's one-time ``_raw_<name>`` variables out of ``variables``.

    Monty keeps a variable bound after the snippet ends, and it has no
    ``del``. Each raw name becomes a call to a host function returning the
    value instead, and a host function's name is gone once the snippet
    ends. Returns the rewritten code and the functions to look up.
    """
    raw = {name for name in variables if name.startswith(RAW_PREFIX)}
    if not raw:
        return code, {}
    lookups = {}
    for name in raw:
        value = to_monty(variables.pop(name))
        lookups[name] = lambda value=value: value
    return _call_names(code, raw), lookups


def _call_names(code: str, names: set[str]) -> str:
    """``code`` with each standalone use of ``names`` turned into a call.
    Attribute names, strings, and comments are left alone."""
    lines = code.splitlines(keepends=True)
    edits: list[tuple[int, int]] = []  # (line index, end column) of each name to call
    previous = None
    try:
        for token in tokenize.generate_tokens(io.StringIO(code).readline):
            if (
                token.type == tokenize.NAME
                and token.string in names
                and not (previous is not None and previous.type == tokenize.OP and previous.string == ".")
            ):
                edits.append((token.end[0] - 1, token.end[1]))
            if token.type not in (tokenize.NL, tokenize.COMMENT):
                previous = token
    except (tokenize.TokenError, SyntaxError):
        return code  # Monty reports the syntax error itself
    for line, column in reversed(edits):
        lines[line] = lines[line][:column] + "()" + lines[line][column:]
    return "".join(lines)
