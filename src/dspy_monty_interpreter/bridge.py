"""Bridge between dspy's sandbox session (dspy 3.5+) and Monty.

dspy 3.5 makes sub-agents and non-JSON inputs available in an RLM's REPL by
running a bootstrap in the interpreter. Monty cannot run that bootstrap, so
the adapter answers it on the host. dspy then treats the interpreter as one
that shares the host's memory: it rebuilds inputs on the host and runs
sub-agents on the host's dspy. Monty's ``ClassInstance`` carries the
resulting host objects into the sandbox, and an ``ExternalModule`` named
``dspy`` lets sandbox code build sub-agents that run on the host.

Host objects cross with the narrowest exposure that works. An input that is
not plain data exposes only its public data attributes; a caller who wants
the sandbox to call methods wraps the object in ``ClassInstance`` with an
explicit ``allowed_methods``. A sub-agent exposes only its call, so sandbox
code cannot reach module methods such as ``save()`` or ``load()``.
"""

from __future__ import annotations

import datetime
import decimal
import inspect
import io
import pathlib
import tokenize
from collections.abc import Callable, Mapping
from typing import Any

from dspy.primitives.prediction import Prediction
from pydantic_monty import ClassInstance, ClassType, ExternalModule

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

PROXY_INSTRUCTIONS = (
    "Inputs that are not plain data are proxies for objects outside the sandbox: "
    "repr(x) shows their data attributes, which you can read."
)
SUB_AGENT_INSTRUCTIONS = (
    "In the sandbox, `import dspy` provides only Predict, ChainOfThought, ReAct, and RLM. "
    "They run outside the sandbox. ReAct accepts only the provided tools; "
    "functions you define in the REPL cannot be passed to it."
)

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


def answer_bootstrap(variables: Mapping[str, Any]) -> None:
    """Register a host namespace for the session, as the bootstrap does in a
    process that shares the host's memory."""
    _facade._in_host[variables["__dspy_spec"]["session"]] = {}


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


class HostModule(ClassInstance):
    """A dspy module on the host that sandbox code can only call. Its
    attributes and other methods, such as ``save()``, stay on the host."""

    def method_allowed(self, name: str) -> bool:
        return name == "__call__"

    def call_method(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if name != "__call__":
            raise self.attr_error(name)
        return to_monty(self.value(*args, **kwargs))


def dspy_module(tools: Mapping[str, Callable[..., Any]]) -> ExternalModule:
    """The ``dspy`` module sandbox code imports. ``tools`` is the
    interpreter's live tool mapping; ReAct resolves tool names against it."""
    import dspy

    def builder(module_cls: type) -> Callable[..., HostModule]:
        def build(signature: Any, **kwargs: Any) -> HostModule:
            return HostModule(module_cls(signature, **kwargs))

        build.__name__ = module_cls.__name__
        return build

    def react(signature: Any, tools: list[Any] | None = None, **kwargs: Any) -> HostModule:
        return HostModule(dspy.ReAct(signature, tools=[resolve_tool(t) for t in tools or []], **kwargs))

    def resolve_tool(tool: Any) -> Callable[..., Any]:
        # A provided tool reaches the host as its name; a REPL function as a repr string.
        available = {name: fn for name, fn in tools.items() if not is_bridge_tool(name)}
        if isinstance(tool, str) and tool in available:
            return available[tool]
        raise TypeError(
            f"ReAct tools must be the provided tools ({', '.join(sorted(available)) or 'none'}); got {tool!r}"
        )

    react.__name__ = "ReAct"
    return ExternalModule(
        {
            "Predict": builder(dspy.Predict),
            "ChainOfThought": builder(dspy.ChainOfThought),
            "ReAct": react,
            "RLM": builder(dspy.RLM),
        }
    )


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
