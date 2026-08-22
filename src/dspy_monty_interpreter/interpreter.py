"""MontyInterpreter: DSPy CodeInterpreter backed by Monty."""

from __future__ import annotations

import inspect
import os
import re
import threading
from collections import Counter
from typing import Any, Callable, Literal

from dspy.primitives.code_interpreter import (
    CodeExecutionError,
    CodeInterpreterError,
    FinalOutput,
)
from dspy.utils.callback import BaseCallback, with_callbacks
from pydantic_monty import (
    AbstractOS,
    Monty,
    MontyCrashedError,
    MontyRuntimeError,
    MontySession,
    MontySyntaxError,
    MountDir,
    ResourceLimits,
)

_SUBMIT_HALT_MESSAGE = "SUBMIT() called; execution stopped"


class _SubmitHalt(Exception):
    """Raised inside SUBMIT() to stop the sandbox at the call site."""


# Matches markdown code fences wrapping the entire code string.
_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:\s*(?:python|py)\s*)?\n(.*?)```\s*$",
    re.DOTALL | re.IGNORECASE,
)


_BASE_EXECUTION_INSTRUCTIONS = (
        "Python runs in Monty, a sandboxed interpreter written in Rust, not CPython. "
        "State persists across executions: variables, functions, and classes defined in one "
        "step are available in later steps. "
        "Nothing is pre-imported; write the import statements you need. "
        "Only these standard-library modules can be imported: re, json, math, datetime, "
        "collections, itertools, dataclasses, typing, sys, os, asyncio, unicodedata. "
        "Other stdlib modules (decimal, statistics, csv, functools, operator, ...) and all "
        "third-party packages (numpy, pandas, requests, ...) are NOT available; there is no pip. "
        "There is no network access, no subprocesses, and no environment variables. "
        "Supported syntax includes classes, decorators, @dataclass, comprehensions, "
        "generators, with statements, and try/except; match statements are NOT supported. "
        "Methods of built-in types cannot be passed as values: max(d, key=d.get) and "
        "sorted(items, key=str.lower) fail with AttributeError. Call them instead, e.g. "
        "max(d, key=lambda k: d[k]) or max(d.items(), key=lambda kv: kv[1]). "
        "print() output is returned to you; a bare expression on the last line is returned "
        "only when nothing was printed. "
        "SUBMIT() stops execution immediately, so print and inspect results in an earlier "
        "step before calling it."
    )

_NO_FILESYSTEM_INSTRUCTIONS = "The filesystem is unavailable: no directories are mounted. "

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
             "and open(path); os.path and os.walk are not available):"]
    for m in mount_list:
        mode = _MOUNT_MODE_DESCRIPTIONS.get(m.mode, m.mode)
        line = f"- {m.virtual_path} ({mode})"
        if listing_limit > 0:
            line += f" containing {_list_host_dir(m.host_path, listing_limit)}"
        lines.append(line)
    return "\n".join(lines) + "\n"


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


def _build_execution_instructions(
    mounts: MountDir | list[MountDir] | None, listing_limit: int = _MOUNT_LISTING_LIMIT
) -> str:
    return _BASE_EXECUTION_INSTRUCTIONS + _describe_mounts(mounts, listing_limit)


class _MontyFactory:
    """Zero-argument interpreter factory that carries execution_instructions.

    A bare ``lambda: MontyInterpreter(...)`` has no ``execution_instructions``
    attribute, so RLM would put nothing about Monty in its prompt.
    """

    def __init__(self, kwargs: dict[str, Any]) -> None:
        self._kwargs = kwargs
        self.execution_instructions = _build_execution_instructions(
            kwargs.get("mounts"), kwargs.get("mount_listing_limit", _MOUNT_LISTING_LIMIT)
        )

    def __call__(self) -> MontyInterpreter:
        return MontyInterpreter(**self._kwargs)

    def __repr__(self) -> str:
        return f"MontyInterpreter.factory({self._kwargs!r})"


class MontyInterpreter:
    """DSPy CodeInterpreter implementation backed by Monty.

    Monty is a secure Python interpreter written in Rust. Unlike the default
    PythonInterpreter (Deno/Pyodide), Monty has no WASM bootstrap and provides
    strict sandboxing with no network or environment access. As of Monty
    0.0.19, code runs in a pool of ``monty`` worker subprocesses: a crashed
    or timed-out worker is replaced transparently without taking down the
    host process. Filesystem access can be enabled per-interpreter via the
    ``mounts`` parameter.

    Pass ``resource_limits`` (a ``pydantic_monty.ResourceLimits`` dict) to
    cap each ``execute()`` call. ``max_memory`` (bytes) is enforced by
    Monty's allocator as of 0.0.20, so a runaway allocation raises
    ``MemoryError`` in the sandbox rather than on the host; it surfaces as
    ``CodeExecutionError`` and the session keeps its state. Other keys:
    ``max_duration_secs``, ``max_recursion_depth``, ``gc_interval``.

    State persists across ``execute()`` calls via ``MontySession``, Monty's
    incremental REPL session — each snippet is compiled and run against
    the persistent heap and namespace without replaying prior snippets.

    Sessions are per-thread: concurrent ``execute()`` calls from different
    threads (e.g. ``dspy.Evaluate`` / ``dspy.Parallel`` running the same RLM
    with ``num_threads``) each get their own isolated session, all sharing
    one worker pool. RLM's per-forward reset discards only the calling
    thread's session. For thread counts above your CPU count, pass
    ``max_processes`` so the pool has one worker per thread.

    Usage with RLM::

        # DSPy creates and shuts down one interpreter per forward():
        rlm = dspy.RLM("context -> answer", interpreter_factory=MontyInterpreter)
        result = rlm(context="...")

        # Or pass a caller-owned interpreter positionally and reuse it:
        interpreter = MontyInterpreter(request_timeout=10.0)
        result = rlm(interpreter, context="...")

    ``execution_instructions`` is read off the factory by RLM and added to
    the action prompt, so the model knows Monty's stdlib and syntax limits.
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
    ) -> None:
        self._tools: dict[str, Callable[..., str]] = dict(tools) if tools else {}
        self.output_fields: list[dict] | None = output_fields
        self.__tools_registered: bool = False
        self._resource_limits: ResourceLimits | None = resource_limits
        self._mounts: MountDir | list[MountDir] | None = mounts
        self._os_access: AbstractOS | None = os_access
        self._request_timeout: float | None = request_timeout
        self._max_processes: int | None = max_processes
        self._pool: Monty | None = None
        self._lock = threading.Lock()
        self._generation: int = 0
        self._thread_local = threading.local()
        # Sessions from every thread, so shutdown() can reclaim them all.
        self._live_sessions: dict[int, MontySession] = {}
        # Instance-level DSPy callbacks; dspy.settings.callbacks also apply.
        self.callbacks: list[BaseCallback] = list(callbacks or [])
        # Instance copy that also describes this interpreter's mounts.
        self.execution_instructions: str = _build_execution_instructions(mounts, mount_listing_limit)

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
                    self._pool = Monty(
                        request_timeout=self._request_timeout,
                        max_processes=self._max_processes,
                    )
                    self._pool.__enter__()
                pool = self._pool
            session = pool.checkout(limits=self._resource_limits)
            session.__enter__()
            local.session = session
            with self._lock:
                self._live_sessions[id(session)] = session
        return local.session

    def _discard_session(self) -> None:
        """Return the calling thread's worker to the pool; the thread's
        next execute() checks out a fresh session with empty state."""
        session = getattr(self._thread_local, "session", None)
        if session is not None:
            with self._lock:
                self._live_sessions.pop(id(session), None)
            try:
                session.__exit__(None, None, None)
            except Exception:
                pass
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
        return self._tools[tool_name](**kwargs)

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
        variables = variables or {}
        code = _strip_code_fences(code)

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
            name: self._make_external_fn(name) for name in self._tools
        }
        external_fns["SUBMIT"] = submit_fn

        session = self._ensure_session()
        try:
            result = session.feed_run(
                code,
                inputs=variables if variables else None,
                external_lookup=external_fns,
                print_callback=print_callback,
                mount=self._mounts,
                os=self._os_access,
            )
        except MontySyntaxError as e:
            raise SyntaxError(str(e)) from e
        except MontyCrashedError as e:
            # The worker died (crash or request_timeout); the session and
            # its state are lost. Discard it so the next execute() checks
            # out a fresh session, but still honor a captured SUBMIT.
            self._discard_session()
            if submit_box:
                args, kwargs = submit_box[0]
                return _handle_submit(args, kwargs, self.output_fields)
            raise CodeInterpreterError(str(e)) from e
        except MontyRuntimeError as e:
            # SUBMIT halts by raising; honor the box whether the error is
            # that halt or a genuine error that fired after SUBMIT.
            if submit_box:
                self._has_state = True
                args, kwargs = submit_box[0]
                return _handle_submit(args, kwargs, self.output_fields)
            raise CodeExecutionError(e.display("type-msg")) from e

        self._has_state = True

        if submit_box:
            args, kwargs = submit_box[0]
            return _handle_submit(args, kwargs, self.output_fields)

        return _build_output(result, print_output)

    @with_callbacks
    def shutdown(self) -> None:
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
                pass
        if pool is not None:
            try:
                pool.__exit__(None, None, None)
            except Exception:
                pass
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
        return FinalOutput(dict(zip(field_names, args)))
    if len(args) == 1:
        return FinalOutput(args[0])
    if len(args) == 0:
        return FinalOutput(None)
    return FinalOutput(args[0])


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
