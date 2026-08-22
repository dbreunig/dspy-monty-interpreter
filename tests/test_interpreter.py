"""Tests for MontyInterpreter."""

import pytest
from dspy.primitives.code_interpreter import (
    CodeInterpreter,
    CodeInterpreterError,
    FinalOutput,
)

from dspy_monty_interpreter import MontyInterpreter, MountDir

# --- Protocol conformance ---


def test_implements_protocol():
    interp = MontyInterpreter()
    assert isinstance(interp, CodeInterpreter)


# --- Basic execution ---


def test_expression():
    interp = MontyInterpreter()
    result = interp.execute("1 + 2")
    assert result == "3"


def test_no_output():
    interp = MontyInterpreter()
    result = interp.execute("x = 42", variables={"x": 0})
    assert result is None


def test_print_capture():
    interp = MontyInterpreter()
    result = interp.execute('print("hello")')
    assert result == "hello"


def test_print_multiline():
    interp = MontyInterpreter()
    result = interp.execute('print("a")\nprint("b")')
    assert result == "a\nb"


def test_variable_injection():
    interp = MontyInterpreter()
    result = interp.execute("x + y", variables={"x": 10, "y": 32})
    assert result == "42"


# --- State persistence ---


def test_variable_persists_across_calls():
    interp = MontyInterpreter()
    interp.execute("x = 42")
    result = interp.execute("x + 8")
    assert result == "50"


def test_function_persists_across_calls():
    interp = MontyInterpreter()
    interp.execute("def double(n):\n    return n * 2")
    result = interp.execute("double(21)")
    assert result == "42"


def test_closure_persists_across_calls():
    interp = MontyInterpreter()
    interp.execute('prefix = "Answer: "')
    interp.execute("def fmt(text):\n    return prefix + text")
    result = interp.execute('fmt("42")')
    assert result == "Answer: 42"


def test_multiple_accumulations():
    interp = MontyInterpreter()
    interp.execute("a = 1")
    interp.execute("b = a + 1")
    interp.execute("c = a + b")
    result = interp.execute("a + b + c")
    assert result == "6"


# --- SUBMIT handling ---


def test_submit_kwargs():
    interp = MontyInterpreter()
    result = interp.execute('SUBMIT(answer="42")')
    assert isinstance(result, FinalOutput)
    assert result.output == {"answer": "42"}


def test_submit_single_positional():
    interp = MontyInterpreter()
    result = interp.execute("SUBMIT(42)")
    assert isinstance(result, FinalOutput)
    assert result.output == 42


def test_submit_positional_with_output_fields():
    interp = MontyInterpreter(output_fields=[{"name": "answer"}, {"name": "confidence"}])
    result = interp.execute('SUBMIT("yes", 0.9)')
    assert isinstance(result, FinalOutput)
    assert result.output == {"answer": "yes", "confidence": 0.9}


def test_submit_single_positional_with_one_output_field():
    # SUBMIT(value) with a single declared output field should map the
    # value to that field, not return it bare. Otherwise RLM rejects
    # with "FINAL returned <type>, expected dict with fields: [...]".
    interp = MontyInterpreter(output_fields=[{"name": "quarterly_findings"}])
    result = interp.execute("SUBMIT([1, 2, 3])")
    assert isinstance(result, FinalOutput)
    assert result.output == {"quarterly_findings": [1, 2, 3]}


def test_submit_dict_matching_schema_passes_through():
    # If the model already builds the schema dict and submits it,
    # don't double-wrap.
    interp = MontyInterpreter(output_fields=[{"name": "answer"}])
    result = interp.execute('SUBMIT({"answer": "42"})')
    assert isinstance(result, FinalOutput)
    assert result.output == {"answer": "42"}


def test_submit_no_args():
    interp = MontyInterpreter()
    result = interp.execute("SUBMIT()")
    assert isinstance(result, FinalOutput)
    assert result.output is None


def test_submit_after_state_accumulation():
    interp = MontyInterpreter()
    interp.execute("x = 10")
    interp.execute("y = x * 2")
    result = interp.execute("SUBMIT(answer=x + y)")
    assert isinstance(result, FinalOutput)
    assert result.output == {"answer": 30}


# --- Tool dispatch ---


def test_tool_call():
    call_log = []

    def my_tool(query: str) -> str:
        call_log.append(query)
        return "tool result"

    interp = MontyInterpreter(tools={"my_tool": my_tool})
    result = interp.execute('my_tool(query="hello")')
    assert result == "tool result"
    assert call_log == ["hello"]


def test_tool_call_positional_args():
    # LLM-emitted code commonly calls tools positionally — the wrapper
    # must accept *args, not just **kwargs.
    call_log = []

    def merge(a: str, b: str) -> str:
        call_log.append((a, b))
        return f"{a}+{b}"

    interp = MontyInterpreter(tools={"merge": merge})
    assert interp.execute('merge("x", "y")') == "x+y"
    assert interp.execute('merge("x", b="y")') == "x+y"
    assert call_log == [("x", "y"), ("x", "y")]


def test_tool_result_used_in_code():
    def lookup(key: str) -> str:
        return "found_value"

    interp = MontyInterpreter(tools={"lookup": lookup})
    result = interp.execute('result = lookup(key="x")\nprint(result)')
    assert result == "found_value"


def test_tool_error_propagation():
    def failing_tool() -> str:
        raise ValueError("tool broke")

    interp = MontyInterpreter(tools={"failing_tool": failing_tool})
    with pytest.raises(CodeInterpreterError, match="ValueError"):
        interp.execute("failing_tool()")


def test_tool_error_caught_by_code():
    def failing_tool() -> str:
        raise ValueError("oops")

    interp = MontyInterpreter(tools={"failing_tool": failing_tool})
    result = interp.execute(
        "try:\n    failing_tool()\nexcept ValueError:\n    print('caught')"
    )
    assert result == "caught"


# --- Tool isolation across calls ---


def test_tool_called_once_across_accumulations():
    call_count = [0]

    def counted_tool(x: str) -> str:
        call_count[0] += 1
        return f"result_{call_count[0]}"

    interp = MontyInterpreter(tools={"counted_tool": counted_tool})
    interp.execute('a = counted_tool(x="first")')
    assert call_count[0] == 1

    # MontyRepl persists the bound value of `a` natively — the earlier
    # tool call is not re-invoked because old code never re-runs.
    result = interp.execute("print(a)")
    assert call_count[0] == 1  # NOT called again
    assert result == "result_1"


def test_tool_caching_with_new_calls():
    call_log = []

    def my_tool(x: str) -> str:
        call_log.append(x)
        return f"got_{x}"

    interp = MontyInterpreter(tools={"my_tool": my_tool})
    interp.execute('a = my_tool(x="first")')
    assert call_log == ["first"]

    interp.execute('b = my_tool(x="second")')
    # "first" is not re-called — MontyRepl persists `a` natively.
    assert call_log == ["first", "second"]

    result = interp.execute("print(a + ' ' + b)")
    assert result == "got_first got_second"
    assert call_log == ["first", "second"]


# --- Print suppression during replay ---


def test_old_prints_not_repeated():
    interp = MontyInterpreter()
    result1 = interp.execute('print("first")')
    assert result1 == "first"

    result2 = interp.execute('print("second")')
    assert result2 == "second"  # NOT "first\nsecond"


# --- Error mapping ---


def test_syntax_error():
    interp = MontyInterpreter()
    with pytest.raises(SyntaxError):
        interp.execute("def")


def test_runtime_error():
    interp = MontyInterpreter()
    with pytest.raises(CodeInterpreterError):
        interp.execute("1 / 0")


def test_name_error():
    interp = MontyInterpreter()
    with pytest.raises(CodeInterpreterError):
        interp.execute("undefined_var")


# --- Error recovery ---


def test_failed_code_not_accumulated():
    """MontyRepl preserves partial mutations from failed snippets (Python REPL
    semantics). In this test the error occurs before any mutation, so x is
    unchanged. Note: if the snippet were 'x = 99\\n1/0', x would be 99 after
    the failure — unlike the old replay architecture which would revert to 10."""
    interp = MontyInterpreter()
    interp.execute("x = 10")

    with pytest.raises(CodeInterpreterError):
        interp.execute("undefined_var")

    result = interp.execute("x + 5")
    assert result == "15"


# --- RLM compatibility ---


def test_tools_update_mutates():
    interp = MontyInterpreter()
    interp.tools.update({"new_tool": lambda: "hi"})
    assert "new_tool" in interp.tools


def test_output_fields_settable():
    interp = MontyInterpreter()
    interp.output_fields = [{"name": "answer"}]
    assert interp.output_fields == [{"name": "answer"}]


def test_tools_registered_settable():
    interp = MontyInterpreter()
    interp._tools_registered = True
    assert interp._tools_registered is True
    interp._tools_registered = False
    assert interp._tools_registered is False


def test_tools_registered_reset_clears_state():
    """Setting _tools_registered = False (as RLM does between forward()
    calls) should clear accumulated interpreter state."""
    interp = MontyInterpreter()
    interp.execute("x = 42")

    # Simulate what RLM does at the start of each forward() call
    interp._tools_registered = False  # triggers reset (code_history is non-empty)

    with pytest.raises(CodeInterpreterError):
        interp.execute("x")  # x should no longer exist


def test_tools_registered_no_reset_when_clean():
    """Setting _tools_registered = False on a fresh interpreter should NOT
    clear state — there's nothing to clear."""
    interp = MontyInterpreter()

    # No code has been executed, so this is a no-op
    interp._tools_registered = False

    # Interpreter should still work normally
    result = interp.execute("1 + 1")
    assert result == "2"


# --- Lifecycle ---


def test_context_manager():
    with MontyInterpreter() as interp:
        result = interp.execute("1 + 1")
        assert result == "2"


def test_start_shutdown_idempotent():
    interp = MontyInterpreter()
    interp.start()
    interp.start()
    interp.execute("1 + 1")
    interp.shutdown()
    interp.shutdown()


def test_shutdown_clears_state():
    interp = MontyInterpreter()
    interp.execute("x = 42")
    interp.shutdown()
    with pytest.raises(CodeInterpreterError):
        interp.execute("x")


# --- Cross-forward() / SUBMIT persistence ---


def test_submit_replay_does_not_retrigger():
    """After SUBMIT, a new execute() picks up native REPL state without
    re-triggering the old SUBMIT. State simply persists — there is no replay."""
    call_count = [0]

    def my_tool(prompt: str) -> str:
        call_count[0] += 1
        return f"response_{call_count[0]}"

    interp = MontyInterpreter(tools={"my_tool": my_tool})

    # Simulate forward() #1: two iterations, second calls SUBMIT
    interp.execute('data = my_tool(prompt="q1")')
    assert call_count[0] == 1

    result = interp.execute("SUBMIT(answer=data)")
    assert isinstance(result, FinalOutput)
    assert result.output == {"answer": "response_1"}

    # Simulate forward() #2: state still persists, and new code runs against
    # that state with no re-execution of prior snippets.
    result2 = interp.execute('new_data = my_tool(prompt="q2")\nprint(data + " " + new_data)')
    assert call_count[0] == 2
    assert result2 == "response_1 response_2"


def test_tool_not_recalled_after_submit():
    """Old tools are not re-invoked because MontyRepl persists state natively
    (not because they are cached)."""
    call_log = []

    def llm_query(prompt: str) -> str:
        call_log.append(prompt)
        return f"answer_for_{prompt}"

    interp = MontyInterpreter(tools={"llm_query": llm_query})

    # forward() #1
    interp.execute('x = llm_query(prompt="first")')
    interp.execute("SUBMIT(answer=x)")
    assert call_log == ["first"]

    # forward() #2 — old llm_query call never happens; only "second" is new.
    result = interp.execute('y = llm_query(prompt="second")\nprint(x + " " + y)')
    assert call_log == ["first", "second"]
    assert result == "answer_for_first answer_for_second"


def test_state_persists_across_submit_boundaries():
    """Variables from before SUBMIT should be available after SUBMIT."""
    interp = MontyInterpreter(tools={"my_tool": lambda: "val"})

    interp.execute("a = 1")
    interp.execute("b = a + 1")
    interp.execute("SUBMIT(answer=b)")

    # After SUBMIT, a and b should still be accessible
    result = interp.execute("a + b")
    assert result == "3"


def test_tool_changes_between_calls():
    """Persisted state still works when tools dict is replaced between calls."""
    def tool_v1(x: str) -> str:
        return "v1"

    def tool_v2(x: str) -> str:
        return "v2"

    interp = MontyInterpreter(tools={"my_tool": tool_v1})
    interp.execute('a = my_tool(x="test")')

    # Replace tools entirely (simulates RLM creating fresh tools)
    interp._tools.clear()
    interp._tools["my_tool"] = tool_v2
    interp._tools["new_tool"] = lambda: "new"

    # Old code never re-runs under MontyRepl, so `a` retains v1's return value
    # regardless of the new tool mapping.
    result = interp.execute("print(a)")
    assert result == "v1"


# --- Code fence stripping ---


def test_strip_python_code_fence():
    interp = MontyInterpreter()
    result = interp.execute("```python\n1 + 2\n```")
    assert result == "3"


def test_strip_py_code_fence():
    interp = MontyInterpreter()
    result = interp.execute("```py\n1 + 2\n```")
    assert result == "3"


def test_strip_bare_code_fence():
    interp = MontyInterpreter()
    result = interp.execute("```\n1 + 2\n```")
    assert result == "3"


def test_no_strip_inline_backticks():
    """Backticks that aren't wrapping the entire code should be left alone."""
    interp = MontyInterpreter()
    result = interp.execute("x = 'hello'\nprint(x)")
    assert result == "hello"


# --- Filesystem mounts ---


def test_mount_read_only():
    """Sandboxed code can read files from a read-only mount."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "data.txt").write_text("hello from mount")
        interp = MontyInterpreter(
            mounts=MountDir(virtual_path="/data", host_path=tmpdir, mode="read-only")
        )
        result = interp.execute(
            "from pathlib import Path\nPath('/data/data.txt').read_text()"
        )
        assert result == "hello from mount"


def test_mount_overlay_write():
    """Overlay mount captures writes in memory without modifying the host."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "original.txt").write_text("original")
        interp = MontyInterpreter(
            mounts=MountDir(virtual_path="/data", host_path=tmpdir, mode="overlay")
        )
        result = interp.execute(
            "from pathlib import Path\n"
            "Path('/data/new.txt').write_text('created')\n"
            "Path('/data/new.txt').read_text()"
        )
        assert result == "created"
        # Host filesystem not modified
        assert not Path(tmpdir, "new.txt").exists()


def test_mount_read_only_blocks_write():
    """Read-only mount rejects write operations."""
    import tempfile
    interp = MontyInterpreter(
        mounts=MountDir(virtual_path="/data", host_path=tempfile.mkdtemp(), mode="read-only")
    )
    with pytest.raises(CodeInterpreterError):
        interp.execute(
            "from pathlib import Path\nPath('/data/file.txt').write_text('x')"
        )


def test_mount_overlay_discarded_across_executes():
    """Overlay writes are per-execute (per feed) as of Monty 0.0.19 —
    they are discarded when the execute() call ends."""
    import tempfile
    interp = MontyInterpreter(
        mounts=MountDir(virtual_path="/data", host_path=tempfile.mkdtemp(), mode="overlay")
    )
    interp.execute(
        "from pathlib import Path\nPath('/data/state.txt').write_text('transient')"
    )
    with pytest.raises(CodeInterpreterError, match="FileNotFoundError"):
        interp.execute("Path('/data/state.txt').read_text()")


def test_mount_read_write_persists_across_executes():
    """read-write mounts write through to the host, so state survives
    across execute() calls (the replacement for overlay persistence)."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        interp = MontyInterpreter(
            mounts=MountDir(virtual_path="/data", host_path=tmpdir, mode="read-write")
        )
        interp.execute(
            "from pathlib import Path\nPath('/data/state.txt').write_text('persisted')"
        )
        result = interp.execute("Path('/data/state.txt').read_text()")
        assert result == "persisted"
        assert Path(tmpdir, "state.txt").read_text() == "persisted"


# --- SUBMIT / error edge cases ---


def test_submit_honored_despite_post_submit_error():
    """If code calls SUBMIT then later errors, SUBMIT result is still returned."""
    interp = MontyInterpreter()
    result = interp.execute('SUBMIT(answer="got it")\n1/0')
    assert isinstance(result, FinalOutput)
    assert result.output == {"answer": "got it"}


def test_partial_mutation_persists_on_error():
    """Monty's REPL session preserves partial mutations from failed
    snippets, matching Python REPL semantics."""
    interp = MontyInterpreter()
    interp.execute("x = 1")
    with pytest.raises(CodeInterpreterError):
        interp.execute("x = 99\n1/0")  # x is set before the error
    result = interp.execute("x")
    assert result == "99"  # not reverted to 1


# --- DSPy interpreter lifecycle callbacks (dspy >= 3.3.1) ---


def _lifecycle_recorder():
    """A BaseCallback that records every on_interpreter_* event."""
    from dspy.utils.callback import BaseCallback

    events = []

    class Recorder(BaseCallback):
        def on_interpreter_execute_start(self, call_id, instance, inputs):
            events.append(("execute_start", call_id, instance, inputs))

        def on_interpreter_execute_end(self, call_id, outputs, exception=None):
            events.append(("execute_end", call_id, outputs, exception))

        def on_interpreter_tool_call_start(self, call_id, instance, inputs):
            events.append(("tool_call_start", call_id, instance, inputs))

        def on_interpreter_tool_call_end(self, call_id, outputs, exception=None):
            events.append(("tool_call_end", call_id, outputs, exception))

        def on_interpreter_startup_start(self, call_id, instance, inputs):
            events.append(("startup_start", call_id, instance, inputs))

        def on_interpreter_startup_end(self, call_id, outputs, exception=None):
            events.append(("startup_end", call_id, outputs, exception))

        def on_interpreter_shutdown_start(self, call_id, instance, inputs):
            events.append(("shutdown_start", call_id, instance, inputs))

        def on_interpreter_shutdown_end(self, call_id, outputs, exception=None):
            events.append(("shutdown_end", call_id, outputs, exception))

        # Tool-level events are the job of dspy.Tool (RLM wraps user tools
        # in one). The interpreter must not fire them itself.
        def on_tool_start(self, call_id, instance, inputs):
            events.append(("tool_start", call_id, instance, inputs))

        def on_tool_end(self, call_id, outputs, exception=None):
            events.append(("tool_end", call_id, outputs, exception))

    return Recorder(), events


def test_execute_fires_interpreter_execute_callbacks():
    import dspy

    recorder, events = _lifecycle_recorder()
    interp = MontyInterpreter()
    with dspy.context(callbacks=[recorder]):
        result = interp.execute("print(1 + 1)")

    assert result == "2"
    kinds = [e[0] for e in events]
    assert kinds == ["execute_start", "execute_end"]
    _, start_id, instance, inputs = events[0]
    assert instance is interp
    assert inputs["code"] == "print(1 + 1)"
    _, end_id, outputs, exc = events[1]
    assert end_id == start_id
    assert outputs == "2"
    assert exc is None


def test_execute_end_callback_receives_exception():
    import dspy
    from dspy.primitives.code_interpreter import CodeExecutionError

    recorder, events = _lifecycle_recorder()
    interp = MontyInterpreter()
    with dspy.context(callbacks=[recorder]), pytest.raises(CodeExecutionError):
        interp.execute("1/0")

    assert events[-1][0] == "execute_end"
    assert isinstance(events[-1][3], CodeExecutionError)


def test_tool_call_fires_interpreter_tool_call_callbacks_once():
    """A sandbox->host tool call fires on_interpreter_tool_call_* exactly
    once, nested under the execute call, and fires NO on_tool_* events."""
    import dspy

    recorder, events = _lifecycle_recorder()

    def search(query: str) -> str:
        return f"result for {query}"

    interp = MontyInterpreter(tools={"search": search})
    with dspy.context(callbacks=[recorder]):
        result = interp.execute('search(query="python")')

    assert result == "result for python"
    kinds = [e[0] for e in events]
    assert kinds == ["execute_start", "tool_call_start", "tool_call_end", "execute_end"]
    assert "tool_start" not in kinds

    _, tc_id, instance, inputs = events[1]
    assert instance is interp
    assert inputs == {"tool_name": "search", "kwargs": {"query": "python"}}
    _, tc_end_id, outputs, exc = events[2]
    assert tc_end_id == tc_id
    assert outputs == "result for python"
    assert exc is None


def test_tool_call_end_callback_receives_exception():
    import dspy
    from dspy.primitives.code_interpreter import CodeExecutionError

    recorder, events = _lifecycle_recorder()

    def bad_tool() -> str:
        raise ValueError("boom")

    interp = MontyInterpreter(tools={"bad_tool": bad_tool})
    with dspy.context(callbacks=[recorder]), pytest.raises(CodeExecutionError):
        interp.execute("bad_tool()")

    tc_end = [e for e in events if e[0] == "tool_call_end"]
    assert len(tc_end) == 1
    assert tc_end[0][2] is None
    assert isinstance(tc_end[0][3], ValueError)


def test_tool_call_active_call_id_nests_under_execute():
    """Inside a tool, ACTIVE_CALL_ID is the tool-call id, whose parent is
    the execute call id; it is restored afterwards."""
    import dspy
    from dspy.utils.callback import ACTIVE_CALL_ID

    recorder, events = _lifecycle_recorder()
    captured = []

    def spy_tool() -> str:
        captured.append(ACTIVE_CALL_ID.get())
        return "ok"

    interp = MontyInterpreter(tools={"spy_tool": spy_tool})
    with dspy.context(callbacks=[recorder]):
        interp.execute("spy_tool()")

    tool_call_id = next(e for e in events if e[0] == "tool_call_start")[1]
    assert captured == [tool_call_id]
    assert ACTIVE_CALL_ID.get() is None


def test_start_and_shutdown_fire_lifecycle_callbacks():
    import dspy

    recorder, events = _lifecycle_recorder()
    interp = MontyInterpreter()
    with dspy.context(callbacks=[recorder]):
        interp.start()
        interp.shutdown()

    kinds = [e[0] for e in events]
    assert kinds == ["startup_start", "startup_end", "shutdown_start", "shutdown_end"]
    assert events[0][2] is interp


def test_instance_level_callbacks_are_honored():
    """Callbacks passed to the constructor fire without dspy.context()."""
    recorder, events = _lifecycle_recorder()
    interp = MontyInterpreter(callbacks=[recorder])
    assert interp.execute("print('hi')") == "hi"
    assert [e[0] for e in events] == ["execute_start", "execute_end"]


def test_tool_no_callbacks_fast_path():
    """Tools work normally when no callbacks are configured."""
    call_log = []

    def my_tool(x: str) -> str:
        call_log.append(x)
        return "ok"

    interp = MontyInterpreter(tools={"my_tool": my_tool})
    result = interp.execute('my_tool(x="test")')
    assert result == "ok"
    assert call_log == ["test"]


def test_tools_can_be_swapped_between_executes():
    """RLM replaces interpreter.tools entries between forward() calls."""

    def tool_v1(x: str) -> str:
        return "v1"

    def tool_v2(x: str) -> str:
        return "v2"

    interp = MontyInterpreter(tools={"my_tool": tool_v1})
    assert interp.execute('my_tool(x="a")') == "v1"
    interp.tools["my_tool"] = tool_v2
    interp._tools_registered = False
    assert interp.execute('my_tool(x="b")') == "v2"


# --- Error classes (dspy >= 3.3.0) ---


def test_runtime_error_is_code_execution_error():
    """Errors in submitted code are recoverable: RLM only feeds
    CodeExecutionError (not bare CodeInterpreterError) back to the LM."""
    from dspy.primitives.code_interpreter import CodeExecutionError

    interp = MontyInterpreter()
    with pytest.raises(CodeExecutionError, match="ZeroDivisionError"):
        interp.execute("1/0")
    # Session survives a recoverable error.
    interp.execute("x = 1")
    assert interp.execute("x") == "1"


def test_tool_error_is_code_execution_error():
    from dspy.primitives.code_interpreter import CodeExecutionError

    def bad_tool() -> str:
        raise ValueError("boom")

    interp = MontyInterpreter(tools={"bad_tool": bad_tool})
    with pytest.raises(CodeExecutionError, match="boom"):
        interp.execute("bad_tool()")


def test_memory_limit_error_is_code_execution_error():
    from dspy.primitives.code_interpreter import CodeExecutionError
    from pydantic_monty import ResourceLimits

    interp = MontyInterpreter(resource_limits=ResourceLimits(max_memory=50_000_000))
    with pytest.raises(CodeExecutionError, match="MemoryError"):
        interp.execute("y = [0] * 100_000_000")


def test_worker_crash_is_terminal_code_interpreter_error():
    """A dead worker (timeout) is NOT recoverable: bare CodeInterpreterError,
    not CodeExecutionError, so RLM aborts instead of retrying."""
    from dspy.primitives.code_interpreter import CodeExecutionError

    interp = MontyInterpreter(request_timeout=0.5)
    with pytest.raises(CodeInterpreterError) as info:
        interp.execute("while True:\n    pass")
    assert not isinstance(info.value, CodeExecutionError)


# --- SUBMIT halts execution ---


def test_submit_halts_execution():
    """Nothing after SUBMIT() runs: no prints, no assignments."""
    interp = MontyInterpreter()
    interp.execute("x = 1")
    result = interp.execute("SUBMIT(answer=x)\nprint('after')\nx = 2")
    assert result == FinalOutput({"answer": 1})
    assert interp.execute("x") == "1"


def test_submit_halts_inside_loop():
    interp = MontyInterpreter()
    result = interp.execute(
        "hits = []\nfor i in range(10):\n    hits.append(i)\n    if i == 2:\n        SUBMIT(answer=i)"
    )
    assert result == FinalOutput({"answer": 2})
    assert interp.execute("print(hits)") == "[0, 1, 2]"


# --- execution_instructions (dspy >= 3.3.1) ---


def test_execution_instructions_is_class_attribute_string():
    """RLM reads it off the factory (the class), not an instance."""
    assert isinstance(MontyInterpreter.execution_instructions, str)
    assert MontyInterpreter.execution_instructions.strip()


def test_execution_instructions_describe_monty_limitations():
    text = MontyInterpreter.execution_instructions
    assert "match" in text
    assert "State persists" in text
    for mod in ("re", "json", "math", "datetime", "collections", "itertools", "dataclasses"):
        assert f"`{mod}`" in text or f" {mod}," in text or f" {mod}." in text or f" {mod} " in text
    assert "pip" in text or "third-party" in text
    assert "network" in text
    # Monty cannot pass builtin methods as values (max(d, key=d.get) fails).
    assert "key=d.get" in text
    assert "decimal" in text


# --- pydantic-monty 0.0.18 regressions ---


def test_comprehension_store_in_repl():
    """REPL store of names bound inside comprehension expressions (monty #297).
    Previously, building a comprehension that referenced persisted state and
    assigning the result could fail under MontyRepl."""
    interp = MontyInterpreter()
    interp.execute("nums = [1, 2, 3, 4]")
    assert interp.execute("squares = [n * n for n in nums]\nprint(sum(squares))") == "30"
    # The comprehension result persists and is reusable on the next call.
    assert interp.execute("len(squares)") == "4"
    # Dict comprehension referencing prior state behaves the same.
    assert interp.execute("m = {n: n * n for n in nums}\nprint(m[3])") == "9"


def test_context_manager_with_open():
    """`with` / context-manager support plus the sandboxed open() builtin
    (monty #462, #456, #461), exercised over an overlay mount. Overlay
    writes are per-execute, so the write/read round-trip happens in one
    execute() call."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "in.txt").write_text("line one\nline two\n")
        interp = MontyInterpreter(
            mounts=MountDir(virtual_path="/data", host_path=tmpdir, mode="overlay")
        )
        assert interp.execute(
            "with open('/data/in.txt') as f:\n    data = f.read()\nprint(len(data))"
        ) == "18"
        assert interp.execute(
            "with open('/data/out.txt', 'w') as f:\n    f.write('hello cm')\n"
            "with open('/data/out.txt') as f:\n    print(f.read())"
        ) == "hello cm"


def test_parallel_execute_thread_isolation():
    """Concurrent execute() calls from different threads get isolated
    sessions: each thread's variables are its own."""
    import threading

    interp = MontyInterpreter()
    barrier = threading.Barrier(2)
    results: dict[str, str] = {}
    errors: list[Exception] = []

    def worker(name: str, value: int) -> None:
        try:
            barrier.wait()
            interp.execute(f"x = {value}")
            barrier.wait()
            results[name] = interp.execute("print(x)")
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=worker, args=("a", 1)),
        threading.Thread(target=worker, args=("b", 2)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert results == {"a": "1", "b": "2"}
    interp.shutdown()


def test_parallel_reset_only_clears_calling_thread():
    """RLM's per-forward reset (_tools_registered = False) discards only
    the calling thread's session, not other threads' state."""
    import threading

    interp = MontyInterpreter()
    interp.execute("x = 'main'")

    def other_forward() -> None:
        # Simulates RLM starting a forward() on another thread.
        interp._tools_registered = False
        interp.execute("x = 'other'")

    t = threading.Thread(target=other_forward)
    t.start()
    t.join()

    # Main thread's state survived the other thread's reset.
    assert interp.execute("print(x)") == "main"
    interp.shutdown()


def test_parallel_evaluate_smoke():
    """Many sequential tasks spread over a thread pool, each doing an
    RLM-style reset + stateful execute sequence, all come back correct."""
    from concurrent.futures import ThreadPoolExecutor

    interp = MontyInterpreter(max_processes=4)

    def task(i: int) -> str:
        interp._tools_registered = False  # RLM does this per forward()
        interp.execute(f"v = {i}")
        interp.execute("v = v * 10")
        return interp.execute("print(v)")

    with ThreadPoolExecutor(max_workers=4) as ex:
        outputs = list(ex.map(task, range(12)))

    assert outputs == [str(i * 10) for i in range(12)]
    interp.shutdown()


def test_shutdown_reclaims_all_thread_sessions():
    """shutdown() discards sessions created by other threads, and the
    interpreter remains usable afterward with fresh state."""
    import threading

    interp = MontyInterpreter()
    ready = threading.Event()
    release = threading.Event()

    def worker() -> None:
        interp.execute("y = 'thread'")
        ready.set()
        release.wait()

    t = threading.Thread(target=worker)
    t.start()
    ready.wait()
    interp.execute("y = 'main'")
    interp.shutdown()
    release.set()
    t.join()

    # Fresh session after shutdown: y is gone.
    with pytest.raises(CodeInterpreterError):
        interp.execute("y")
    interp.shutdown()


def test_user_defined_class():
    """User-defined classes work as of Monty 0.0.19, including state
    that persists across execute() calls."""
    interp = MontyInterpreter()
    interp.execute(
        "class Counter:\n"
        "    def __init__(self):\n"
        "        self.n = 0\n"
        "    def bump(self):\n"
        "        self.n += 1\n"
        "        return self.n\n"
        "c = Counter()"
    )
    assert interp.execute("c.bump()\nc.bump()\nprint(c.bump())") == "3"


def test_max_memory_limit_raises_and_keeps_state():
    """A max_memory violation surfaces as CodeInterpreterError (MemoryError
    in the sandbox) and, unlike a worker crash, the session keeps its state."""
    from pydantic_monty import ResourceLimits

    interp = MontyInterpreter(resource_limits=ResourceLimits(max_memory=50_000_000))
    interp.start()
    try:
        interp.execute("x = 42")
        with pytest.raises(CodeInterpreterError, match="MemoryError"):
            interp.execute("y = [0] * 100_000_000")
        assert interp.execute("x") == "42"
    finally:
        interp.shutdown()


def test_worker_crash_raises_and_recovers():
    """A crashed worker (here: exceeding request_timeout) surfaces as
    CodeInterpreterError, and the interpreter recovers with a fresh
    session on the next execute()."""
    interp = MontyInterpreter(request_timeout=0.5)
    interp.execute("x = 1")
    with pytest.raises(CodeInterpreterError):
        interp.execute("i = 0\nwhile True:\n    i += 1")
    # Session was lost with the worker; the next call gets a fresh one.
    result = interp.execute("print('alive')")
    assert result == "alive"


def test_external_function_identity():
    """Name-based identity and equality for external function values
    (monty #458) — covers both injected tools and SUBMIT."""
    def my_tool(x: str) -> str:
        return "v"

    interp = MontyInterpreter(tools={"my_tool": my_tool})
    assert interp.execute("f = my_tool\nprint(f is my_tool)") == "True"
    assert interp.execute("f = my_tool\nprint(f == my_tool)") == "True"
    assert interp.execute("print(SUBMIT is SUBMIT)") == "True"


# --- RLM integration (no LM needed) ---


def test_rlm_accepts_class_as_interpreter_factory():
    import dspy

    assert isinstance(MontyInterpreter(), dspy.CodeInterpreter)
    rlm = dspy.RLM("q -> a", interpreter_factory=MontyInterpreter)
    assert "match statements are NOT supported" in rlm.generate_action.signature.instructions


def test_rlm_feeds_runtime_errors_back_to_the_lm():
    """RLM._execute_code turns CodeExecutionError into an '[Error] ...' string
    (a correction turn) but lets a terminal CodeInterpreterError propagate."""
    import dspy

    rlm = dspy.RLM("q -> a", interpreter_factory=MontyInterpreter)
    interp = MontyInterpreter(request_timeout=0.5)
    try:
        result = rlm._execute_code(interp, "1/0", {})
        assert isinstance(result, str) and result.startswith("[Error]")
        assert "ZeroDivisionError" in result

        with pytest.raises(CodeInterpreterError):
            rlm._execute_code(interp, "while True:\n    pass", {})
    finally:
        interp.shutdown()


# --- MontyInterpreter.factory() and conditional execution_instructions ---


def test_factory_returns_configured_instances():
    factory = MontyInterpreter.factory(request_timeout=1.5)
    interp = factory()
    assert isinstance(interp, MontyInterpreter)
    assert interp._request_timeout == 1.5
    assert factory() is not interp


def test_factory_carries_execution_instructions():
    """A plain lambda has no execution_instructions, so RLM would fall back
    to an empty prompt section; factory() must expose one."""
    factory = MontyInterpreter.factory()
    assert factory.execution_instructions == MontyInterpreter.execution_instructions
    assert "Monty" in factory.execution_instructions


def test_instructions_without_mounts_say_filesystem_unavailable():
    text = MontyInterpreter.factory().execution_instructions
    assert "filesystem is unavailable" in text
    assert "Mounted" not in text


def test_instructions_with_mounts_describe_paths_modes_and_contents(tmp_path):
    reports = tmp_path / "reports"
    (reports / "q3").mkdir(parents=True)
    (reports / "readme.md").write_text("notes")
    (reports / "q3" / "forecast.txt").write_text("forecast")
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    factory = MontyInterpreter.factory(
        mounts=[
            MountDir(host_path=str(reports), virtual_path="/data", mode="read-only"),
            MountDir(host_path=str(scratch), virtual_path="/scratch", mode="overlay"),
        ]
    )
    text = factory.execution_instructions
    assert "filesystem is unavailable" not in text
    assert "/data (read-only)" in text
    assert "q3/" in text and "readme.md" in text
    assert "/scratch (overlay" in text and "discarded" in text
    assert "(empty)" in text
    assert "os.listdir" in text and "Path" in text
    assert "os.walk" in text and "os.path" in text  # named as unavailable
    # The factory's instances carry the same text.
    assert factory().execution_instructions == text


def _mount_line(text: str, virtual_path: str) -> str:
    return next(line for line in text.splitlines() if line.startswith(f"- {virtual_path} "))


def test_instructions_small_dir_lists_every_entry(tmp_path):
    for i in range(20):
        (tmp_path / f"f{i:02d}.txt").write_text("x")
    line = _mount_line(
        MontyInterpreter.factory(mounts=MountDir(host_path=str(tmp_path), virtual_path="/d", mode="read-only")).execution_instructions,
        "/d",
    )
    assert "f00.txt" in line and "f19.txt" in line


def test_instructions_large_dir_is_summarized_not_listed(tmp_path):
    for i in range(40):
        (tmp_path / f"f{i:02d}.csv").write_text("x")
    for i in range(3):
        (tmp_path / f"notes{i}.md").write_text("x")
    (tmp_path / "sub").mkdir()
    line = _mount_line(
        MontyInterpreter.factory(mounts=MountDir(host_path=str(tmp_path), virtual_path="/d", mode="read-only")).execution_instructions,
        "/d",
    )
    assert "43 files and 1 directory" in line
    assert ".csv ×40" in line and ".md ×3" in line
    assert "e.g. f00.csv" in line
    assert "f39.csv" not in line
    assert len(line) < 320


def test_instructions_truncate_long_names(tmp_path):
    (tmp_path / ("a" * 120 + ".txt")).write_text("x")
    line = _mount_line(
        MontyInterpreter.factory(mounts=MountDir(host_path=str(tmp_path), virtual_path="/d", mode="read-only")).execution_instructions,
        "/d",
    )
    assert "a" * 120 not in line
    assert "…" in line
    assert len(line) < 120


def test_instructions_respect_character_budget_per_mount(tmp_path):
    # 20 entries of 40 chars each would be ~850 chars as a plain list.
    for i in range(20):
        (tmp_path / (f"{i:02d}_" + "x" * 37)).write_text("x")
    line = _mount_line(
        MontyInterpreter.factory(mounts=MountDir(host_path=str(tmp_path), virtual_path="/d", mode="read-only")).execution_instructions,
        "/d",
    )
    assert len(line) <= 320
    assert "more" in line


def test_instructions_listing_can_be_disabled(tmp_path):
    (tmp_path / "secret.txt").write_text("x")
    text = MontyInterpreter.factory(
        mounts=MountDir(host_path=str(tmp_path), virtual_path="/d", mode="read-only"),
        mount_listing_limit=0,
    ).execution_instructions
    assert "- /d (read-only)" in text
    assert "secret.txt" not in text
    assert "containing" not in text


def test_instructions_listing_limit_is_configurable(tmp_path):
    for i in range(5):
        (tmp_path / f"f{i}.txt").write_text("x")
    text = MontyInterpreter(
        mounts=MountDir(host_path=str(tmp_path), virtual_path="/d", mode="read-only"),
        mount_listing_limit=3,
    ).execution_instructions
    assert "5 files" in text and "f4.txt" not in text


def test_instructions_scan_is_bounded(tmp_path):
    import dspy_monty_interpreter.interpreter as mod

    for i in range(mod._MOUNT_SCAN_LIMIT + 5):
        (tmp_path / f"{i}.txt").write_text("x")
    line = _mount_line(
        MontyInterpreter.factory(mounts=MountDir(host_path=str(tmp_path), virtual_path="/d", mode="read-only")).execution_instructions,
        "/d",
    )
    assert f"{mod._MOUNT_SCAN_LIMIT:,}+ entries" in line


def test_instructions_tolerate_unreadable_host_path(tmp_path):
    """MountDir requires the host path to exist, but it can vanish before
    the factory is built; describe the mount anyway."""
    later = tmp_path / "later"
    later.mkdir()
    mount = MountDir(host_path=str(later), virtual_path="/out", mode="read-write")
    later.rmdir()
    text = MontyInterpreter.factory(mounts=mount).execution_instructions
    assert "/out (read-write" in text
    assert "not readable" in text


def test_rlm_prompt_includes_mount_description(tmp_path):
    import dspy

    (tmp_path / "a.csv").write_text("x")
    factory = MontyInterpreter.factory(
        mounts=MountDir(host_path=str(tmp_path), virtual_path="/data", mode="read-only")
    )
    rlm = dspy.RLM("q -> a", interpreter_factory=factory)
    instructions = rlm.generate_action.signature.instructions
    assert "/data (read-only)" in instructions and "a.csv" in instructions
