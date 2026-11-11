"""Interpreter lifecycle and state across RLM forward() calls."""

import pytest
from dspy.primitives.code_interpreter import (
    CodeInterpreterError,
    FinalOutput,
)

from dspy_monty_interpreter import MontyInterpreter

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
