"""Tests for MontyInterpreter."""

import pytest
from dspy.primitives.code_interpreter import CodeInterpreter, CodeInterpreterError, FinalOutput

from dspy_monty_interpreter import MontyInterpreter, MountDirectory


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
            mounts=MountDirectory("/data", tmpdir, mode="read-only")
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
            mounts=MountDirectory("/data", tmpdir, mode="overlay")
        )
        interp.execute(
            "from pathlib import Path\nPath('/data/new.txt').write_text('created')"
        )
        result = interp.execute("Path('/data/new.txt').read_text()")
        assert result == "created"
        # Host filesystem not modified
        assert not Path(tmpdir, "new.txt").exists()


def test_mount_read_only_blocks_write():
    """Read-only mount rejects write operations."""
    import tempfile
    interp = MontyInterpreter(
        mounts=MountDirectory("/data", tempfile.mkdtemp(), mode="read-only")
    )
    with pytest.raises(CodeInterpreterError):
        interp.execute(
            "from pathlib import Path\nPath('/data/file.txt').write_text('x')"
        )


def test_mount_persists_across_executes():
    """Overlay state persists across execute() calls."""
    import tempfile
    interp = MontyInterpreter(
        mounts=MountDirectory("/data", tempfile.mkdtemp(), mode="overlay")
    )
    interp.execute(
        "from pathlib import Path\nPath('/data/state.txt').write_text('persisted')"
    )
    result = interp.execute("Path('/data/state.txt').read_text()")
    assert result == "persisted"


# --- SUBMIT / error edge cases ---


def test_submit_honored_despite_post_submit_error():
    """If code calls SUBMIT then later errors, SUBMIT result is still returned."""
    interp = MontyInterpreter()
    result = interp.execute('SUBMIT(answer="got it")\n1/0')
    assert isinstance(result, FinalOutput)
    assert result.output == {"answer": "got it"}


def test_partial_mutation_persists_on_error():
    """MontyRepl preserves partial mutations from failed snippets,
    matching Python REPL semantics."""
    interp = MontyInterpreter()
    interp.execute("x = 1")
    with pytest.raises(CodeInterpreterError):
        interp.execute("x = 99\n1/0")  # x is set before the error
    result = interp.execute("x")
    assert result == "99"  # not reverted to 1
