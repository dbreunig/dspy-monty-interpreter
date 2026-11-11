"""MontyInterpreter: DSPy CodeInterpreter backed by Monty."""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import logging
import os
import re
import threading
from collections import ChainMap, Counter
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from dspy.primitives.code_interpreter import (
    CodeExecutionError,
    CodeInterpreterError,
    FinalOutput,
)
from dspy.utils.callback import BaseCallback, with_callbacks
from pydantic_monty import (
    AbstractOS,
    AsyncMontyRemote,
    ExternalModule,
    Monty,
    MontyConnectError,
    MontyCrashedError,
    MontyDisconnectError,
    MontyRefusedError,
    MontyRuntimeError,
    MontySession,
    MontyShutdown,
    MontySyntaxError,
    MontyTypingError,
    MountDir,
    OSPolicy,
    ResourceLimits,
)

from dspy_monty_interpreter import bridge, remote, rewrite, telemetry, typecheck

logger = logging.getLogger(__name__)

_SUBMIT_HALT_MESSAGE = "SUBMIT() called; execution stopped"


class _SubmitHalt(Exception):
    """Raised inside SUBMIT() to stop the sandbox at the call site."""


# Matches markdown code fences wrapping the entire code string.
_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:\s*(?:python|py)\s*)?\n(.*?)```\s*$",
    re.DOTALL | re.IGNORECASE,
)


# Standard-library modules Monty can import. tests/test_interpreter.py checks
# each claim against the installed Monty.
_AVAILABLE_STDLIB = (
    "re", "json", "math", "datetime", "time", "random", "collections", "itertools",
    "functools", "dataclasses", "copy", "base64", "binascii", "typing", "sys", "os",
    "os.path", "pathlib", "asyncio", "unicodedata", "csv", "tomllib", "hashlib",
    "bisect", "decimal", "fnmatch", "annotationlib",
)
# Common stdlib modules Monty lacks, named so the model stops reaching for them.
_UNAVAILABLE_STDLIB_EXAMPLES = ("statistics", "io", "operator", "string", "heapq", "enum")

_BASE_EXECUTION_INSTRUCTIONS = (
    "Python runs in Monty, a sandboxed interpreter written in Rust, not CPython. "
    "State persists across executions: variables, functions, and classes defined in one "
    "step are available in later steps. "
    "Nothing is pre-imported; write the import statements you need. "
    f"Only these standard-library modules can be imported: {', '.join(_AVAILABLE_STDLIB)}. "
    f"Other stdlib modules ({', '.join(_UNAVAILABLE_STDLIB_EXAMPLES)}, ...) and all "
    "third-party packages (numpy, pandas, requests, ...) are NOT available; there is no pip. "
    "There is no network access, no subprocesses, and no environment variables; "
    "datetime.now() and time.time() do work. "
    "Supported syntax includes classes, decorators, @dataclass, complex numbers, comprehensions, "
    "generator expressions, lambdas, with statements, try/except, and match statements; "
    "generator functions (yield) are NOT supported, nor are class inheritance "
    "(including subclassing Exception), or a callable replacement in re.sub(). "
    "Methods of built-in types cannot be passed as values: max(d, key=d.get) and "
    "sorted(items, key=str.lower) fail with AttributeError. Call them instead, e.g. "
    "max(d, key=lambda k: d[k]) or max(d.items(), key=lambda kv: kv[1]). "
    "print() output is returned to you; a bare expression on the last line is returned "
    "only when nothing was printed. "
    "SUBMIT() stops execution immediately, so print and inspect results in an earlier "
    "step before calling it."
)

_NO_FILESYSTEM_INSTRUCTIONS = "The filesystem is unavailable: no directories are mounted."

_DEFAULT_OS_POLICY: OSPolicy = {"sleep": "zero"}

# Monty's own default of 1000 host calls per session is too low for an RLM run
# that loops a tool over its data. Each call costs about 80µs, so this cap
# still stops a runaway loop within a couple of minutes.
_DEFAULT_MAX_SUSPENSIONS = 1_000_000
_SUSPENSION_LIMIT_RE = re.compile(r"suspension limit (\d+) exceeded")

_MOUNT_MODE_DESCRIPTIONS = {
    "read-only": "read-only",
    "overlay": "overlay: writable, but writes are discarded when each execution ends",
    "read-write": "read-write: writes reach the host",
}

_MOUNT_LISTING_LIMIT = 20  # entries listed by name before switching to a summary
_MOUNT_SCAN_LIMIT = 5_000  # directory entries examined before giving up counting
_MOUNT_NAME_LIMIT = 40  # characters of a single entry name
_MOUNT_LINE_BUDGET = 300  # characters for one mount's "containing ..." text
_MOUNT_EXAMPLE_COUNT = 3  # sample names shown in a summary


def _describe_mounts(mounts: MountDir | list[MountDir] | None, listing_limit: int) -> str:
    """Conditional filesystem section for execution_instructions."""
    if mounts is None:
        return _NO_FILESYSTEM_INSTRUCTIONS
    mount_list = [mounts] if isinstance(mounts, MountDir) else list(mounts)
    if not mount_list:
        return _NO_FILESYSTEM_INSTRUCTIONS
    lines = ["Mounted directories (explore with os.listdir(path), pathlib.Path(path).iterdir(), "
             "os.walk(path), and open(path)):"]
    for m in mount_list:
        mode = _MOUNT_MODE_DESCRIPTIONS.get(m.mode, m.mode)
        line = f"- {m.virtual_path} ({mode})"
        if listing_limit > 0:
            line += f" containing {_list_host_dir(m.host_path, listing_limit)}"
        lines.append(line)
    return "\n".join(lines)


def _short_name(entry: os.DirEntry) -> str:
    name = entry.name
    if len(name) > _MOUNT_NAME_LIMIT:
        name = name[: _MOUNT_NAME_LIMIT - 1] + "…"
    return name + "/" if entry.is_dir() else name


def _list_host_dir(host_path: str, listing_limit: int) -> str:
    """Bounded description of ``host_path``'s top level at factory-creation time.

    Small directories are listed by name. Larger ones get a summary (counts,
    extension histogram, a few example names) so a mount with thousands of
    files costs the prompt the same as one with twenty.
    """
    entries: list[os.DirEntry] = []
    truncated = False
    try:
        with os.scandir(host_path) as it:
            for entry in it:
                if len(entries) >= _MOUNT_SCAN_LIMIT:
                    truncated = True
                    break
                entries.append(entry)
    except OSError:
        return "(not readable from the host)"
    if not entries:
        return "(empty)"
    entries.sort(key=lambda e: e.name)

    if len(entries) <= listing_limit and not truncated:
        text = _fit_names([_short_name(e) for e in entries], _MOUNT_LINE_BUDGET)
        if text is not None:
            return text
    return _summarize_entries(entries, truncated)


def _fit_names(names: list[str], budget: int) -> str | None:
    """Join names within ``budget`` chars, eliding the tail; None if even the
    elided form would be mostly ellipsis."""
    text = ", ".join(names)
    if len(text) <= budget:
        return text
    kept: list[str] = []
    for name in names:
        candidate = ", ".join([*kept, name])
        if len(candidate) + len(", and 99 more") > budget:
            break
        kept.append(name)
    if len(kept) < _MOUNT_EXAMPLE_COUNT:
        return None
    return ", ".join(kept) + f", and {len(names) - len(kept)} more"


def _summarize_entries(entries: list[os.DirEntry], truncated: bool) -> str:
    files = [e for e in entries if not e.is_dir()]
    dirs = [e for e in entries if e.is_dir()]
    if truncated:
        head = f"{len(entries):,}+ entries"
    else:
        head = (f"{len(files):,} file{'s' if len(files) != 1 else ''} and "
                f"{len(dirs):,} director{'ies' if len(dirs) != 1 else 'y'}")
    details: list[str] = []
    exts = Counter(os.path.splitext(e.name)[1] or "(no extension)" for e in files)
    if exts:
        details.append(", ".join(f"{ext} ×{n}" for ext, n in exts.most_common(4)))
    examples = [_short_name(e) for e in entries[:_MOUNT_EXAMPLE_COUNT]]
    details.append("e.g. " + ", ".join(examples))
    text = f"{head} ({'; '.join(details)})"
    return text if len(text) <= _MOUNT_LINE_BUDGET else text[: _MOUNT_LINE_BUDGET - 1] + "…"


_MODULE_STUB_BUDGET = 4_000  # characters of one host module's stubs in the prompt


def _describe_external_modules(modules: Mapping[str, Any] | None) -> str:
    """Section naming the host modules sandbox code may import.

    The sandbox has no dir(), help() or __doc__, so the model learns a
    module's API only from here: its ``.pyi`` stubs when given, otherwise
    its public attribute names and function signatures. A factory is named
    but not called, so the module is still built lazily.
    """
    if not modules:
        return ""
    lines = [
        (
            "Host modules you can import (dir() and help() are unavailable; "
            "rely on these descriptions):"
        )
    ]
    for name, module in modules.items():
        if not isinstance(module, ExternalModule):
            lines.append(f"- {name}")
        elif module.stubs:
            stubs = module.stubs.strip()
            if len(stubs) > _MODULE_STUB_BUDGET:
                stubs = stubs[:_MODULE_STUB_BUDGET] + "\n# ... (truncated)"
            lines.append(f"- {name}:\n```python\n{stubs}\n```")
        else:
            lines.append(f"- {name}: {_describe_module_attrs(module)}")
    return "\n".join(lines)


def _describe_module_attrs(module: ExternalModule) -> str:
    parts: list[str] = []
    for key, value in module.attrs.items():
        if key.startswith("_"):
            continue
        if isinstance(value, ExternalModule):
            parts.append(f"{key} (submodule)")
        elif callable(value):
            try:
                parts.append(f"{key}{inspect.signature(value)}")
            except (TypeError, ValueError):
                parts.append(f"{key}(...)")
        else:
            parts.append(key)
    return ", ".join(parts) or "(no public attributes)"


_TYPE_CHECK_INSTRUCTIONS = (
    "Each code block is type-checked before it runs. A block that fails the check does not run at all, "
    "so fix the reported error and resend the whole block."
)


def _build_execution_instructions(
    mounts: MountDir | list[MountDir] | None,
    listing_limit: int = _MOUNT_LISTING_LIMIT,
    external_modules: Mapping[str, Any] | None = None,
    type_check: bool = False,
    persistent: bool = False,
) -> str:
    sections = [_BASE_EXECUTION_INSTRUCTIONS, _describe_mounts(mounts, listing_limit)]
    if type_check:
        sections.append(_TYPE_CHECK_INSTRUCTIONS)
    if persistent:
        sections.append(_PERSISTENT_STATE_INSTRUCTIONS)
    if modules_text := _describe_external_modules(external_modules):
        sections.append(modules_text)
    sections.append(bridge.instructions())
    return "\n".join(sections)


class SessionState:
    """REPL state carried from one interpreter to the next.

    Pass one to ``MontyInterpreter.factory(state=...)`` to keep variables,
    functions and classes across ``dspy.RLM`` calls, such as the turns of a
    conversation. Each interpreter restores the state when its session
    starts and saves it when it shuts down. Host objects do not carry over:
    proxied inputs and sub-agents built with ``dspy`` must be made again.

    Use one state per conversation. ``data`` holds Monty's serialized
    session; only load bytes this process produced or otherwise trusts.
    """

    def __init__(self, data: bytes | None = None) -> None:
        self._lock = threading.Lock()
        self._data = data

    @property
    def data(self) -> bytes | None:
        with self._lock:
            return self._data

    @data.setter
    def data(self, value: bytes | None) -> None:
        with self._lock:
            self._data = value

    def clear(self) -> None:
        self.data = None

    def __repr__(self) -> str:
        size = len(self.data or b"")
        return f"SessionState({size:,} bytes)" if size else "SessionState(empty)"


_PERSISTENT_STATE_INSTRUCTIONS = (
    "Variables, functions and classes from earlier calls in this conversation persist; "
    "sub-agents and input proxies from earlier calls do not."
)


# True while execute() runs in this context. Interpreters made then belong to
# sub-agents built in the sandbox, which must not share a conversation's state.
_inside_execute: contextvars.ContextVar[bool] = contextvars.ContextVar("dspy_monty_inside_execute", default=False)


class _MontyFactory:
    """Zero-argument interpreter factory that carries execution_instructions.

    A bare ``lambda: MontyInterpreter(...)`` has no ``execution_instructions``
    attribute, so RLM would put nothing about Monty in its prompt.
    """

    def __init__(self, kwargs: dict[str, Any]) -> None:
        self._kwargs = kwargs
        self.execution_instructions = _build_execution_instructions(
            kwargs.get("mounts"),
            kwargs.get("mount_listing_limit", _MOUNT_LISTING_LIMIT),
            kwargs.get("external_modules"),
            kwargs.get("type_check", False),
            kwargs.get("state") is not None,
        )

    def __call__(self) -> MontyInterpreter:
        if _inside_execute.get() and "state" in self._kwargs:
            return MontyInterpreter(**{k: v for k, v in self._kwargs.items() if k != "state"})
        return MontyInterpreter(**self._kwargs)

    def __repr__(self) -> str:
        return f"MontyInterpreter.factory({self._kwargs!r})"


class MontyInterpreter:
    """DSPy CodeInterpreter implementation backed by Monty.

    Monty is a secure Python interpreter written in Rust. Unlike the default
    PythonInterpreter (Deno/Pyodide), Monty has no WASM bootstrap and provides
    strict sandboxing with no network or environment access. Code runs in a
    pool of ``monty`` worker subprocesses: a crashed or timed-out worker is
    replaced transparently without taking down the host process. Filesystem
    access can be enabled per-interpreter via the ``mounts`` parameter.

    Pass ``resource_limits`` (a ``pydantic_monty.ResourceLimits`` dict) to
    cap each ``execute()`` call. ``max_memory`` (bytes) is enforced by
    Monty's allocator, so a runaway allocation raises ``MemoryError`` in the
    sandbox rather than on the host; it surfaces as ``CodeExecutionError``
    and the session keeps its state. ``max_suspensions`` caps the host calls (tools, ``llm_query``, host
    module functions) one session may make; the adapter raises Monty's
    default of 1000 to 1,000,000. ``max_feed_duration_secs`` caps one
    ``execute()`` call and ``max_turn_duration_secs`` caps the sandbox time
    between host calls; Monty makes no guarantees about the heap after
    either fires, so the adapter discards the session and the next
    ``execute()`` starts with empty state. Other keys: ``max_recursion_depth``,
    ``max_total_sleep_secs``, ``gc_interval``.

    Pass ``os_policy`` (a ``pydantic_monty.OSPolicy`` dict) to control the
    sandbox's clock, timezone, sleep, and random seed. The adapter defaults
    ``sleep`` to ``'zero'`` so ``time.sleep()`` returns immediately instead
    of holding a worker; pass ``{'sleep': 'system'}`` to restore real
    waits. Clock overrides on an ``os_access`` object are only consulted
    when ``os_policy`` selects ``'call_host'`` for ``datetime``.

    Pass ``external_modules`` (a mapping of name to
    ``pydantic_monty.ExternalModule``, or to a zero-argument factory that
    returns one) to let sandbox code ``import`` host modules. Callables in
    a module run on the host; return plain data, or wrap objects in
    ``pydantic_monty.ClassInstance``. Unlike ``tools``, these calls do not
    fire DSPy tool callbacks. ``execution_instructions`` describes each
    module from its ``stubs``, or from its attribute names and signatures.

    Pass ``type_check=True`` to type-check each snippet before it runs,
    against stubs built from the tools' signatures, the input variables'
    types, and the sandbox's ``dspy`` namespace. A snippet that fails the
    check raises ``CodeExecutionError`` without running, so the session's
    state is untouched. The checker is strict (``d.get(k) + 1`` fails
    because ``get`` may return ``None``), so it suits models that write
    careful code.

    Pass ``state`` (a ``SessionState``) to carry REPL variables, functions
    and classes from one interpreter to the next, such as across the turns
    of an RLM conversation: the session is restored when it starts and
    saved at ``shutdown()``.

    State persists across ``execute()`` calls via ``MontySession``, Monty's
    incremental REPL session — each snippet is compiled and run against
    the persistent heap and namespace without replaying prior snippets.

    Usage with RLM (DSPy 3.4+)::

        # DSPy creates and shuts down one interpreter per forward():
        rlm = dspy.RLM("context -> answer", interpreter_factory=MontyInterpreter)
        result = rlm(context="...")

        # Configure it through the factory, globally or per call:
        dspy.configure(interpreter_factory=MontyInterpreter.factory(request_timeout=10.0))
        result = rlm(context="...", interpreter_factory=MontyInterpreter.factory(mounts=[...]))

    Starting a worker pool costs a few milliseconds, so one interpreter per
    ``forward()`` is the right default even under ``dspy.Evaluate``. If you
    drive one instance from several threads yourself, each thread gets its
    own isolated session on the shared pool; pass ``max_processes`` when
    the thread count exceeds your CPU count.

    Tools may be coroutine functions: ``invoke_tool`` awaits their result,
    also when ``execute()`` runs inside an event loop (``RLM.acall``).

    ``execution_instructions`` is read off the active factory by RLM on
    every action call and added to the prompt, so the model knows Monty's
    stdlib and syntax limits.
    With ``mounts`` it also describes each mounted directory: path, mode,
    and a bounded view of the top-level contents (names for small
    directories, counts and an extension histogram for large ones; see
    ``mount_listing_limit``, 0 to omit contents entirely).
    """

    # RLM reads this off the interpreter factory and adds it to the action
    # prompt under "Execution environment". The class attribute is the
    # mount-free baseline (for ``interpreter_factory=MontyInterpreter``);
    # ``factory()`` and instances carry a version describing their mounts.
    execution_instructions = _build_execution_instructions(None)

    def __init__(
        self,
        tools: dict[str, Callable[..., str]] | None = None,
        output_fields: list[dict] | None = None,
        resource_limits: ResourceLimits | None = None,
        mounts: MountDir | list[MountDir] | None = None,
        os_access: AbstractOS | None = None,
        request_timeout: float | None = None,
        max_processes: int | None = None,
        callbacks: list[BaseCallback] | None = None,
        mount_listing_limit: int = _MOUNT_LISTING_LIMIT,
        os_policy: OSPolicy | None = None,
        external_modules: Mapping[str, ExternalModule | Callable[[], ExternalModule]] | None = None,
        type_check: bool = False,
        state: SessionState | None = None,
        remote: str | Callable[[], Any] | None = None,
    ) -> None:
        self._tools: dict[str, Callable[..., str]] = dict(tools) if tools else {}
        self.output_fields: list[dict] | None = output_fields
        self.__tools_registered: bool = False
        self._resource_limits: ResourceLimits = {
            "max_suspensions": _DEFAULT_MAX_SUSPENSIONS,
            **(resource_limits or {}),
        }
        self._mounts: MountDir | list[MountDir] | None = mounts
        self._os_access: AbstractOS | None = os_access
        # Sandbox code has nothing to wait for, so sleeping only holds a
        # worker (Monty caps each system sleep at 10s); callers opt back in.
        self._os_policy: OSPolicy = {**_DEFAULT_OS_POLICY, **(os_policy or {})}
        self._request_timeout: float | None = request_timeout
        self._max_processes: int | None = max_processes
        self._type_check = type_check
        self._state = state
        self._remote = remote
        # Monty reads the mapping live at import time, so the user's mapping
        # is chained rather than copied.
        self._external_modules = external_modules
        self._session_modules: Mapping[str, Any] | None = external_modules
        self._bridge: bridge.BridgeContext | None = None
        if bridge.SUPPORTED:
            self._bridge = bridge.BridgeContext(self._tools)
            self._session_modules = ChainMap(
                external_modules or {}, {bridge.HOST_MODULE: lambda: bridge.dspy_module(self._bridge)}
            )
        self._pool: Monty | None = None
        self._lock = threading.Lock()
        self._generation: int = 0
        self._thread_local = threading.local()
        # Sessions from every thread, so shutdown() can reclaim them all.
        self._live_sessions: dict[int, MontySession] = {}
        # Instance-level DSPy callbacks; dspy.settings.callbacks also apply.
        self.callbacks: list[BaseCallback] = list(callbacks or [])
        # Instance copy that also describes this interpreter's mounts.
        self.execution_instructions: str = _build_execution_instructions(
            mounts, mount_listing_limit, external_modules, type_check, state is not None
        )

    @classmethod
    def factory(cls, **kwargs: Any) -> Callable[[], MontyInterpreter]:
        """Return an ``interpreter_factory`` for ``dspy.RLM``.

        Equivalent to ``lambda: MontyInterpreter(**kwargs)`` except the
        returned callable carries ``execution_instructions`` describing this
        configuration (including any mounted directories), which RLM reads
        when building its prompt.
        """
        return _MontyFactory(kwargs)

    # Session state is per-thread; _has_state gates whether an RLM reset
    # needs to discard the calling thread's session.
    @property
    def _has_state(self) -> bool:
        return getattr(self._thread_local, "has_state", False)

    @_has_state.setter
    def _has_state(self, value: bool) -> None:
        self._thread_local.has_state = value

    def _ensure_session(self) -> MontySession:
        """Return the calling thread's live REPL session, lazily spawning
        the shared worker pool and checking out a session on first use."""
        local = self._thread_local
        if getattr(local, "generation", None) != self._generation:
            # shutdown() ran since this thread last executed; its old
            # session was already reclaimed there.
            local.session = None
            local.has_state = False
            local.generation = self._generation
        if getattr(local, "session", None) is None:
            with self._lock:
                if self._pool is None:
                    self._pool = self._make_pool()
                    self._pool.__enter__()
                pool = self._pool
            session = pool.checkout(
                limits=self._resource_limits,
                os_policy=self._os_policy,
                external_modules=self._session_modules,
                type_check=self._type_check,
                type_check_stubs=self._type_stubs() if self._type_check else None,
                type_check_format="concise",
            )
            local.declared = set()  # input variables declared to the type checker
            try:
                session.__enter__()
            except (MontyConnectError, MontyRefusedError, MontyDisconnectError) as e:
                raise CodeInterpreterError(f"Could not open a Monty session: {e}") from e
            try:
                restored = self._restore_state(session)
                if self._bridge is not None:
                    # Defines the sandbox's ``dspy`` namespace in every new session.
                    # The first feed sets the working directory from its mounts.
                    # The type checker learns its names from PRELUDE_STUBS instead.
                    session.feed_run(
                        bridge.PRELUDE, print_callback=lambda *_: None, mount=self._mounts, skip_type_check=True
                    )
            except BaseException as e:
                # Return the worker; the next execute() checks out a fresh session.
                try:
                    session.__exit__(None, None, None)
                except Exception:
                    logger.debug("Returning a Monty session to the pool failed", exc_info=True)
                if isinstance(e, (MontyCrashedError, MontyDisconnectError, MontyShutdown, MontyRefusedError)):
                    raise CodeInterpreterError(f"Could not prepare a Monty session: {e}") from e
                raise
            if restored:
                local.has_state = True
            local.session = session
            with self._lock:
                self._live_sessions[id(session)] = session
        return local.session

    def _make_pool(self) -> Any:
        """A local subprocess pool, or a sync facade over a remote one."""
        if self._remote is None:
            return Monty(request_timeout=self._request_timeout, max_processes=self._max_processes)
        if callable(self._remote):
            return remote.SyncPool(self._remote)
        options: dict[str, Any] = {}
        if self._request_timeout is not None:
            options["request_timeout"] = self._request_timeout
        if self._max_processes is not None:
            options["max_processes"] = self._max_processes
        url = self._remote
        return remote.SyncPool(lambda: AsyncMontyRemote(url, **options))

    def _restore_state(self, session: MontySession) -> bool:
        """Load the saved state into a new session; True if there was any."""
        data = self._state.data if self._state is not None else None
        if not data:
            return False
        try:
            session.load_session(data)
        except Exception:
            logger.warning("Could not restore the saved Monty session; starting empty", exc_info=True)
            return False
        return True

    def _save_state(self, sessions: list[MontySession]) -> None:
        """Save the calling thread's session, or the only one, into the state."""
        if self._state is None or not sessions:
            return
        local = self._thread_local
        own = getattr(local, "session", None) if getattr(local, "generation", None) == self._generation else None
        session = own if own in sessions else sessions[0] if len(sessions) == 1 else None
        if session is None:
            logger.warning("Not saving Monty session state: several threads hold sessions")
            return
        try:
            self._state.data = session.dump()
        except Exception:
            logger.warning("Could not save the Monty session state", exc_info=True)

    def _discard_session(self) -> None:
        """Return the calling thread's worker to the pool; the thread's
        next execute() checks out a fresh session with empty state."""
        if self._bridge is not None:
            self._bridge.cancel_runs(threading.get_ident())
        session = getattr(self._thread_local, "session", None)
        if session is not None:
            with self._lock:
                self._live_sessions.pop(id(session), None)
            try:
                session.__exit__(None, None, None)
            except Exception:
                # The worker is being discarded either way.
                logger.debug("Returning a Monty session to the pool failed", exc_info=True)
            self._thread_local.session = None
        self._thread_local.has_state = False

    @with_callbacks
    def invoke_tool(self, tool_name: str, kwargs: dict[str, Any]) -> Any:
        """Run one host tool on behalf of sandbox code.

        ``@with_callbacks`` routes this through DSPy's
        ``on_interpreter_tool_call_start`` / ``_end`` handlers and nests
        ``ACTIVE_CALL_ID`` under the enclosing ``execute()`` call. Tool-level
        ``on_tool_*`` events are left to ``dspy.Tool`` (RLM wraps every user
        tool in one), so nothing fires twice.
        """
        if tool_name not in self._tools:
            raise CodeInterpreterError(f"Unknown tool: {tool_name}")
        result = self._tools[tool_name](**kwargs)
        return _await_in_sync(result) if inspect.isawaitable(result) else result

    def _make_external_fn(self, tool_name: str) -> Callable[..., Any]:
        """Build the callable Monty invokes for ``tool_name``.

        Positional arguments from the sandbox are bound to parameter names
        so ``invoke_tool`` (and the callbacks it fires) always see kwargs.
        """
        fn = self._tools[tool_name]

        def external(*args: Any, **kwargs: Any) -> Any:
            if args:
                try:
                    bound = inspect.signature(fn).bind_partial(*args, **kwargs)
                except (TypeError, ValueError) as e:
                    raise TypeError(f"{tool_name}(): {e}") from e
                kwargs = dict(bound.arguments)
            return self.invoke_tool(tool_name, kwargs)

        return external

    @property
    def tools(self) -> dict[str, Callable[..., str]]:
        return self._tools

    # RLM sets ``_tools_registered = False`` via _inject_execution_context at
    # the start of every forward() call.  We intercept that write so we can
    # automatically clear REPL state between RLM runs.
    @property  # type: ignore[override]
    def _tools_registered(self) -> bool:
        return self.__tools_registered

    @_tools_registered.setter
    def _tools_registered(self, value: bool) -> None:
        if not value and self._has_state:
            self._discard_session()
        self.__tools_registered = value

    @with_callbacks
    def start(self) -> None:
        self._ensure_session()

    @with_callbacks
    @telemetry.traced_execute
    def execute(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
    ) -> Any:
        """Execute Python code and return the result.

        State from prior successful execute() calls is preserved via
        ``MontySession``'s persistent heap and namespace.

        Returns:
            FinalOutput if SUBMIT() was called, str for print output,
            or None if no output was produced.

        Raises:
            CodeExecutionError: On runtime errors in the code or a tool it
                called. The session survives; RLM feeds these back to the LM.
            CodeInterpreterError: When the worker died or timed out. The
                session and its state are lost; RLM treats this as terminal.
            SyntaxError: On syntax errors.
        """
        token = _inside_execute.set(True)
        try:
            return self._execute(code, variables)
        finally:
            _inside_execute.reset(token)
            if self._bridge is not None:
                # Sandbox code resumes a sub-agent only within the snippet
                # that started it, so a run still paused now is orphaned.
                self._bridge.cancel_runs(threading.get_ident())

    def _execute(self, code: str, variables: dict[str, Any] | None) -> Any:
        variables = dict(variables or {})
        if bridge.is_bootstrap(code, variables):
            self._bridge.namespace = bridge.answer_bootstrap(variables)
            return None
        code = _strip_code_fences(code)
        flex_driver = rewrite.flex_driver(code) if self._bridge is not None else None
        if flex_driver is not None:
            # dspy.Flex runs its module with globals(), which Monty lacks.
            code = rewrite.flex_driver_code(*flex_driver)
            variables = {name: bridge.to_plain(value) for name, value in variables.items()}
        elif self._bridge is not None:
            self._bridge.functions.update(rewrite.functions(code))
            code = rewrite.rewrite(code)
        code, raw_lookups = bridge.take_raw_variables(code, variables)
        variables = {name: bridge.to_monty(value) for name, value in variables.items()}

        print_output: list[str] = []

        def print_callback(_stream: Literal["stdout", "stderr"], text: str) -> None:
            print_output.append(text)

        # SUBMIT captures its args into a box, then raises so Monty stops
        # right there: nothing after the SUBMIT() call runs. The raise
        # surfaces as MontyRuntimeError below, where a non-empty box wins.
        submit_box: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def submit_fn(*args: Any, **kwargs: Any) -> None:
            submit_box.append((args, kwargs))
            raise _SubmitHalt(_SUBMIT_HALT_MESSAGE)

        external_fns: dict[str, Callable[..., Any]] = {
            name: self._make_external_fn(name) for name in self._tools if not bridge.is_bridge_tool(name)
        }
        external_fns.update(self._session_lookups())
        external_fns.update(raw_lookups)
        external_fns["SUBMIT"] = submit_fn

        session = self._ensure_session()
        # Flex's driver and dspy's raw inputs use names only the host knows.
        skip_type_check = flex_driver is not None or bool(raw_lookups)
        try:
            if self._type_check and not skip_type_check:
                self._declare_inputs(session, variables)
            result = session.feed_run(
                code,
                inputs=variables if variables else None,
                external_lookup=external_fns,
                print_callback=print_callback,
                mount=self._mounts,
                os=self._os_access,
                skip_type_check=skip_type_check,
            )
        except MontySyntaxError as e:
            raise SyntaxError(str(e)) from e
        except MontyTypingError as e:
            raise CodeExecutionError(f"Type check failed, so the code did not run:\n{e.display().strip()}") from e
        except MontyCrashedError as e:
            # The worker died (crash or request_timeout); the session and
            # its state are lost. Discard it so the next execute() checks
            # out a fresh session, but still honor a captured SUBMIT.
            self._discard_session()
            if submit_box:
                args, kwargs = submit_box[0]
                return _handle_submit(args, kwargs, self.output_fields)
            raise CodeInterpreterError(str(e)) from e
        except (MontyDisconnectError, MontyRefusedError, MontyShutdown, MontyConnectError) as e:
            # A remote server dropped or refused the session; its state is gone.
            self._discard_session()
            if submit_box:
                args, kwargs = submit_box[0]
                return _handle_submit(args, kwargs, self.output_fields)
            raise CodeInterpreterError(f"{type(e).__name__}: {e}") from e
        except MontyRuntimeError as e:
            # SUBMIT halts by raising; honor the box whether the error is
            # that halt or a genuine error that fired after SUBMIT.
            if submit_box:
                self._has_state = True
                args, kwargs = submit_box[0]
                return _handle_submit(args, kwargs, self.output_fields)
            message = e.display("type-msg")
            if m := _SUSPENSION_LIMIT_RE.search(message):
                message += (
                    f"; the sandbox allows at most {m.group(1)} calls to tools and llm_query per run, "
                    "and every further call will fail. Batch the work instead, e.g. with llm_query_batched."
                )
            elif _is_time_limit(e):
                # A feed or turn time limit stops the sandbox mid-operation
                # and Monty makes no guarantees about its heap afterwards:
                # discard the session so the next execute() starts clean.
                self._discard_session()
                message += "; the sandbox was reset and all variables were lost"
            raise CodeExecutionError(message) from e

        self._has_state = True
        if self._type_check and not skip_type_check:
            self._widen_assigned(session, code)

        if submit_box:
            args, kwargs = submit_box[0]
            return _handle_submit(args, kwargs, self.output_fields)

        if flex_driver is not None:
            return result  # {"fields": ...}, read by dspy.Flex on the host
        return _build_output(result, print_output)

    def _type_stubs(self) -> str:
        """Declarations of every host name sandbox code may use, for the checker."""
        functions = {name: fn for name, fn in self._tools.items() if not bridge.is_bridge_tool(name)}
        names: list[str] = []
        if self._bridge is None:
            return typecheck.stubs(functions)
        if self._bridge.namespace is not None:
            # dspy.Tool wraps the function whose signature the checker needs.
            functions.update({name: getattr(t, "func", t) for name, t in self._bridge.provided_tools().items()})
            names = list(self._bridge.custom_types())
        return typecheck.stubs(functions, names, extra=bridge.PRELUDE_STUBS)

    def _declare_inputs(self, session: MontySession, variables: dict[str, Any]) -> None:
        """Declare input variables the checker has not seen, by type."""
        declared: set[str] = self._thread_local.declared
        new = {name: typecheck.value_annotation(v) for name, v in variables.items() if name not in declared}
        if new:
            session.feed_run(typecheck.declarations(new), print_callback=lambda *_: None)
            declared.update(new)

    def _widen_assigned(self, session: MontySession, code: str) -> None:
        """Redeclare the names ``code`` assigned as ``Any``, so later snippets
        may assign them values of other types."""
        names = typecheck.assigned_names(code)
        if names:
            session.feed_run(
                typecheck.declarations(dict.fromkeys(sorted(names), "Any")), print_callback=lambda *_: None
            )

    def _session_lookups(self) -> dict[str, Callable[..., Any]]:
        """Names a dspy session adds to the sandbox: provided tools that dspy
        serves itself (a Flex's), and the signature's custom types. A custom
        type crosses back to the host as its name, which sub-agents resolve."""
        if self._bridge is None or self._bridge.namespace is None:
            return {}
        lookups: dict[str, Callable[..., Any]] = {}
        for name in self._bridge.session_tools():
            lookups[name] = self._bridge.session_tool(name).func
        for name in self._bridge.custom_types():
            lookups.setdefault(name, _custom_type_stub(name))
        return lookups

    @with_callbacks
    def shutdown(self) -> None:
        if self._bridge is not None:
            self._bridge.cancel_runs()
        with self._lock:
            sessions = list(self._live_sessions.values())
        self._save_state(sessions)
        with self._lock:
            sessions = list(self._live_sessions.values())
            self._live_sessions.clear()
            pool, self._pool = self._pool, None
            # Invalidate every thread's cached session reference.
            self._generation += 1
        for session in sessions:
            try:
                session.__exit__(None, None, None)
            except Exception:
                # The worker is being discarded either way.
                logger.debug("Returning a Monty session to the pool failed", exc_info=True)
        if pool is not None:
            try:
                pool.__exit__(None, None, None)
            except Exception:
                logger.debug("Shutting down the Monty worker pool failed", exc_info=True)
        self.__tools_registered = False

    def __enter__(self) -> MontyInterpreter:
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.shutdown()


def _handle_submit(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    output_fields: list[dict] | None,
) -> FinalOutput:
    """Process SUBMIT() arguments into a FinalOutput."""
    if kwargs:
        return FinalOutput(dict(kwargs))
    if args and output_fields:
        field_names = [f["name"] for f in output_fields]
        # If a single dict positional arg already matches the output
        # schema (e.g. SUBMIT({"answer": x})), pass it through verbatim
        # instead of wrapping it again as {field_name: that_dict}.
        if (
            len(args) == 1
            and isinstance(args[0], dict)
            and set(args[0].keys()).issubset(field_names)
        ):
            return FinalOutput(dict(args[0]))
        # SUBMIT may pass fewer or more values than there are output fields.
        return FinalOutput(dict(zip(field_names, args, strict=False)))
    if len(args) == 1:
        return FinalOutput(args[0])
    if len(args) == 0:
        return FinalOutput(None)
    return FinalOutput(args[0])


def _custom_type_stub(name: str) -> Callable[..., Any]:
    def stub(*_: Any, **__: Any) -> Any:
        raise TypeError(f"{name} values are plain data in the sandbox: build a dict of its fields instead.")

    return stub


def _await_in_sync(awaitable: Any) -> Any:
    """Run an awaitable to completion from synchronous code.

    RLM wraps coroutine tools as ``async def`` and calls ``execute()``
    synchronously even from ``aforward()``, so a loop may already be running
    in this thread; then the awaitable runs on its own loop in a helper
    thread, as DSPy's ``LocalInterpreter`` does.
    """

    async def run() -> Any:
        return await awaitable

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run())
    ctx = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(ctx.run, asyncio.run, run()).result()


def _is_time_limit(e: MontyRuntimeError) -> bool:
    """True when ``max_feed_duration_secs`` or ``max_turn_duration_secs``
    stopped the sandbox, as opposed to code raising ``TimeoutError`` itself."""
    return isinstance(e.exception(), TimeoutError) and "time limit exceeded" in e.display("msg")


def _strip_code_fences(code: str) -> str:
    """Remove markdown code fences wrapping the entire code string."""
    m = _CODE_FENCE_RE.match(code)
    if m:
        return m.group(1)
    return code


def _build_output(output: Any, print_output: list[str]) -> Any:
    """Build the return value from Monty's output and captured prints."""
    captured = "".join(print_output)
    if captured.endswith("\n"):
        captured = captured[:-1]
    if captured:
        return captured
    if output is not None:
        return str(output)
    return None
