"""MontyInterpreter: DSPy CodeInterpreter backed by Monty."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput
from pydantic_monty import (
    Monty,
    MontyComplete,
    MontyFutureSnapshot,
    MontyRuntimeError,
    MontySnapshot,
    MontySyntaxError,
    ResourceLimits,
)

# Sentinel external function name used to separate replayed code from new code.
_BOUNDARY = "__mci_boundary__"

# Matches markdown code fences wrapping the entire code string.
_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:\s*(?:python|py)\s*)?\n(.*?)```\s*$",
    re.DOTALL | re.IGNORECASE,
)


@dataclass
class _CachedCall:
    """A cached external function call result from a previous execution."""

    func_name: str
    result: Any = None
    exception: BaseException | None = None


class MontyInterpreter:
    """DSPy CodeInterpreter implementation backed by Monty.

    Monty is a secure Python interpreter written in Rust. Unlike the default
    PythonInterpreter (Deno/Pyodide), Monty starts in microseconds, has no
    subprocess overhead, and provides strict sandboxing with no filesystem,
    network, or environment access.

    State persists across execute() calls via code accumulation: each call
    re-executes all prior successful code blocks with cached tool results,
    then runs the new code. A boundary marker separates replay from live
    execution so that prints and tool calls from prior blocks are suppressed.

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
    ) -> None:
        self._tools: dict[str, Callable[..., str]] = dict(tools) if tools else {}
        self.output_fields: list[dict] | None = output_fields
        self._tools_registered: bool = False
        self._resource_limits: ResourceLimits | None = resource_limits
        self._code_history: list[str] = []
        self._call_cache: list[_CachedCall] = []
        # Track all external function names ever used in accumulated code,
        # so Monty always recognizes them even if tools change between calls.
        self._ext_fn_history: set[str] = set()

    @property
    def tools(self) -> dict[str, Callable[..., str]]:
        return self._tools

    def start(self) -> None:
        pass

    def execute(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
    ) -> Any:
        """Execute Python code and return the result.

        State from prior successful execute() calls is preserved by
        re-executing accumulated code with cached tool results.

        Returns:
            FinalOutput if SUBMIT() was called, str for print output,
            or None if no output was produced.

        Raises:
            CodeInterpreterError: On runtime errors.
            SyntaxError: On syntax errors.
        """
        variables = variables or {}
        code = _strip_code_fences(code)

        # Build combined code: old blocks + boundary + new code
        has_history = len(self._code_history) > 0
        if has_history:
            old_code = "\n".join(self._code_history)
            full_code = f"{old_code}\n{_BOUNDARY}()\n{code}"
        else:
            full_code = code

        # Determine input and external function names.
        # Include current tools, historical tools (from accumulated code),
        # and SUBMIT. This ensures Monty recognizes all function names that
        # appear anywhere in the accumulated + new code.
        input_names = list(variables.keys()) if variables else None
        ext_fn_names = list(
            {*self._tools.keys(), *self._ext_fn_history, "SUBMIT"}
        )
        if has_history:
            ext_fn_names.append(_BOUNDARY)

        # Parse code
        try:
            m = Monty(
                full_code,
                inputs=input_names,
                external_functions=ext_fn_names,
            )
        except MontySyntaxError as e:
            raise SyntaxError(str(e)) from e
        except MontyRuntimeError as e:
            raise CodeInterpreterError(e.display("type-msg")) from e

        # Print callback that suppresses output during replay
        in_replay = [has_history]
        new_print_output: list[str] = []

        def print_callback(_stream: Literal["stdout"], text: str) -> None:
            if not in_replay[0]:
                new_print_output.append(text)

        # Start execution
        try:
            progress = m.start(
                inputs=variables if variables else None,
                limits=self._resource_limits,
                print_callback=print_callback,
            )
        except MontySyntaxError as e:
            raise SyntaxError(str(e)) from e
        except MontyRuntimeError as e:
            raise CodeInterpreterError(e.display("type-msg")) from e

        # Start/resume loop
        replay_index = 0
        total_cached = len(self._call_cache)
        new_calls: list[_CachedCall] = []

        try:
            while True:
                if isinstance(progress, MontyComplete):
                    # Execution finished — commit history and return
                    self._code_history.append(code)
                    self._call_cache.extend(new_calls)
                    self._ext_fn_history.update(
                        c.func_name for c in new_calls
                    )
                    return _build_output(progress.output, new_print_output)

                if isinstance(progress, MontySnapshot):
                    fn_name = progress.function_name

                    # Boundary marker: switch from replay to live
                    if fn_name == _BOUNDARY:
                        in_replay[0] = False
                        progress = progress.resume(return_value=None)
                        continue

                    # During replay: use cached results for ALL external
                    # calls (tools AND SUBMIT). This avoids re-calling
                    # tools (e.g. llm_query) and re-triggering SUBMIT
                    # from previously accumulated code.
                    if in_replay[0] and replay_index < total_cached:
                        cached = self._call_cache[replay_index]
                        replay_index += 1
                        if cached.exception is not None:
                            progress = progress.resume(exception=cached.exception)
                        else:
                            progress = progress.resume(return_value=cached.result)
                        continue

                    # Live SUBMIT: return FinalOutput and commit history
                    if fn_name == "SUBMIT":
                        # Cache the SUBMIT so it can be replayed in future
                        # execute() calls (resumed with None to continue
                        # past it during replay).
                        new_calls.append(
                            _CachedCall(func_name="SUBMIT", result=None)
                        )
                        self._code_history.append(code)
                        self._call_cache.extend(new_calls)
                        self._ext_fn_history.update(
                            c.func_name for c in new_calls
                        )
                        return _handle_submit(
                            progress.args, progress.kwargs, self.output_fields
                        )

                    # Live tool call
                    if fn_name in self._tools:
                        progress, call = _call_tool(
                            progress, fn_name, self._tools[fn_name]
                        )
                        new_calls.append(call)
                        continue

                    # Unknown function
                    exc = NameError(f"Unknown function: {fn_name}")
                    progress = progress.resume(exception=exc)
                    continue

                if isinstance(progress, MontyFutureSnapshot):
                    raise CodeInterpreterError(
                        "Async execution is not supported by MontyInterpreter."
                    )

                raise CodeInterpreterError(
                    f"Unexpected Monty progress type: {type(progress)}"
                )

        except MontyRuntimeError as e:
            # Don't commit failed code to history
            raise CodeInterpreterError(e.display("type-msg")) from e

    def shutdown(self) -> None:
        self._code_history.clear()
        self._call_cache.clear()
        self._ext_fn_history.clear()
        self._tools_registered = False

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


def _call_tool(
    snapshot: MontySnapshot,
    fn_name: str,
    tool_fn: Callable[..., str],
) -> tuple[MontySnapshot | MontyFutureSnapshot | MontyComplete, _CachedCall]:
    """Call a host-side tool and resume execution."""
    try:
        result = tool_fn(*snapshot.args, **snapshot.kwargs)
    except Exception as exc:
        cached = _CachedCall(func_name=fn_name, exception=exc)
        progress = snapshot.resume(exception=exc)
    else:
        cached = _CachedCall(func_name=fn_name, result=result)
        progress = snapshot.resume(return_value=result)
    return progress, cached


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
