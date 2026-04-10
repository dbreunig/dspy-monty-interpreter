"""MontyInterpreter: DSPy CodeInterpreter backed by Monty."""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Callable, Literal

import dspy
from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput
from dspy.utils.callback import ACTIVE_CALL_ID
from pydantic_monty import (
    MontyRepl,
    MontyRuntimeError,
    MontySyntaxError,
    MountDirectory,
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
    PythonInterpreter (Deno/Pyodide), Monty starts in microseconds, has no
    subprocess overhead, and provides strict sandboxing with no network or
    environment access. Filesystem access can be enabled per-interpreter via
    the ``mounts`` parameter.

    State persists across ``execute()`` calls via ``MontyRepl``, Monty's
    built-in incremental REPL — each snippet is compiled and run against
    the persistent heap and namespace without replaying prior snippets.

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
        mounts: MountDirectory | list[MountDirectory] | None = None,
    ) -> None:
        self._tools: dict[str, Callable[..., str]] = dict(tools) if tools else {}
        self.output_fields: list[dict] | None = output_fields
        self.__tools_registered: bool = False
        self._resource_limits: ResourceLimits | None = resource_limits
        self._mounts: MountDirectory | list[MountDirectory] | None = mounts
        self._repl: MontyRepl = self._new_repl()
        self._has_state: bool = False
        self._tool_instances: dict[str, dspy.Tool] = {}

    def _new_repl(self) -> MontyRepl:
        return MontyRepl(limits=self._resource_limits)

    def _wrap_tool_with_callbacks(self, name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a tool function to fire DSPy on_tool_start/on_tool_end callbacks."""
        def wrapper(**kwargs: Any) -> Any:
            callbacks = dspy.settings.get("callbacks", [])
            if not callbacks:
                return fn(**kwargs)

            # Lazily build and cache a Tool instance for this function
            if name not in self._tool_instances:
                self._tool_instances[name] = dspy.Tool(fn, name=name)
            tool_instance = self._tool_instances[name]

            call_id = uuid.uuid4().hex

            for cb in callbacks:
                try:
                    cb.on_tool_start(call_id=call_id, instance=tool_instance, inputs=kwargs)
                except Exception as e:
                    logging.getLogger(__name__).warning(f"Callback error on tool start: {e}")

            parent_call_id = ACTIVE_CALL_ID.get()
            ACTIVE_CALL_ID.set(call_id)

            result = None
            exception = None
            try:
                result = fn(**kwargs)
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
            self._repl = self._new_repl()
            self._has_state = False
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
        ``MontyRepl``'s persistent heap and namespace.

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

        def print_callback(_stream: Literal["stdout"], text: str) -> None:
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

        try:
            result = self._repl.feed_run(
                code,
                inputs=variables if variables else None,
                external_functions=external_fns,
                print_callback=print_callback,
                mount=self._mounts,
            )
        except MontySyntaxError as e:
            raise SyntaxError(str(e)) from e
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
        self._repl = self._new_repl()
        self._has_state = False
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
    if len(args) > 1 and output_fields:
        field_names = [f["name"] for f in output_fields]
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
