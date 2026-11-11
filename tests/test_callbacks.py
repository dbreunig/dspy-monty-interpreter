"""DSPy interpreter callbacks fired by execute(), tools, and lifecycle methods."""

import pytest

from dspy_monty_interpreter import MontyInterpreter


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
