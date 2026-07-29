"""MontyInterpreter: DSPy CodeInterpreter backed by Monty."""

from __future__ import annotations

import inspect
import logging
import re
import threading
import uuid
from typing import Any, Callable, Literal

import dspy
from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput
from dspy.utils.callback import ACTIVE_CALL_ID
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

# Matches markdown code fences wrapping the entire code string.
_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:\s*(?:python|py)\s*)?\n(.*?)```\s*$",
    re.DOTALL | re.IGNORECASE,
)


class MontyInterpreter:
    """DSPy CodeInterpreter implementation backed by Monty.

    Monty is a secure Python interpreter written in Rust. Unlike the default
    PythonInterpreter (Deno/Pyodide), Monty has no WASM bootstrap and provides
    strict sandboxing with no network or environment access. As of Monty
    0.0.19, code runs in a pool of ``monty`` worker subprocesses: a crashed
    or timed-out worker is replaced transparently without taking down the
    host process. Filesystem access can be enabled per-interpreter via the
    ``mounts`` parameter.

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

        interpreter = MontyInterpreter()
        rlm = dspy.RLM("context -> answer", interpreter=interpreter)
        result = rlm(context="...")
    """

    def __init__(
        self,
        tools: dict[str, Callable[..., str]] | None = None,
        output_fields: list[dict] | None = None,
        resource_limits: ResourceLimits | None = None,
        mounts: MountDir | list[MountDir] | None = None,
        os_access: AbstractOS | None = None,
        request_timeout: float | None = None,
        max_processes: int | None = None,
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
        self._tool_instances: dict[str, dspy.Tool] = {}

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

    def _wrap_tool_with_callbacks(self, name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a tool function to fire DSPy on_tool_start/on_tool_end callbacks."""
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            callbacks = dspy.settings.get("callbacks", [])
            if not callbacks:
                return fn(*args, **kwargs)

            # Lazily build and cache a Tool instance for this function
            if name not in self._tool_instances:
                self._tool_instances[name] = dspy.Tool(fn, name=name)
            tool_instance = self._tool_instances[name]

            # Best-effort: bind positional args to parameter names so the
            # callback `inputs` dict is keyed by name. Fall back to kwargs
            # alone if the signature can't be introspected or bound.
            try:
                bound = inspect.signature(fn).bind_partial(*args, **kwargs)
                inputs: dict[str, Any] = dict(bound.arguments)
            except (TypeError, ValueError):
                inputs = dict(kwargs)

            call_id = uuid.uuid4().hex

            for cb in callbacks:
                try:
                    cb.on_tool_start(call_id=call_id, instance=tool_instance, inputs=inputs)
                except Exception as e:
                    logging.getLogger(__name__).warning(f"Callback error on tool start: {e}")

            parent_call_id = ACTIVE_CALL_ID.get()
            ACTIVE_CALL_ID.set(call_id)

            result = None
            exception = None
            try:
                result = fn(*args, **kwargs)
                return result
            except Exception as e:
                exception = e
                raise
            finally:
                ACTIVE_CALL_ID.set(parent_call_id)
                for cb in callbacks:
                    try:
                        cb.on_tool_end(call_id=call_id, outputs=result, exception=exception)
                    except Exception as e:
                        logging.getLogger(__name__).warning(f"Callback error on tool end: {e}")

        return wrapper

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
        if not value:
            self._tool_instances.clear()
        self.__tools_registered = value

    def start(self) -> None:
        pass

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
            CodeInterpreterError: On runtime errors.
            SyntaxError: On syntax errors.
        """
        variables = variables or {}
        code = _strip_code_fences(code)

        print_output: list[str] = []

        def print_callback(_stream: Literal["stdout", "stderr"], text: str) -> None:
            print_output.append(text)

        # SUBMIT captures its args into a box and returns None so the VM
        # continues executing any code after the SUBMIT() call.
        submit_box: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def submit_fn(*args: Any, **kwargs: Any) -> None:
            submit_box.append((args, kwargs))

        external_fns: dict[str, Callable[..., Any]] = {
            name: self._wrap_tool_with_callbacks(name, fn)
            for name, fn in self._tools.items()
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
            # If SUBMIT was called before the error, honor it.
            if submit_box:
                self._has_state = True
                args, kwargs = submit_box[0]
                return _handle_submit(args, kwargs, self.output_fields)
            raise CodeInterpreterError(e.display("type-msg")) from e

        self._has_state = True

        if submit_box:
            args, kwargs = submit_box[0]
            return _handle_submit(args, kwargs, self.output_fields)

        return _build_output(result, print_output)

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
        self._tool_instances.clear()
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
