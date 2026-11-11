"""Host tools called from sandbox code, sync and async."""

import pytest
from dspy.primitives.code_interpreter import CodeInterpreterError

from dspy_monty_interpreter import MontyInterpreter

from _support import double_async

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


# --- Async tools (dspy >= 3.4 wraps coroutine tools as async def) ---


def test_async_tool_result_is_awaited():
    interp = MontyInterpreter(tools={"double": double_async})
    assert interp.execute("double(21)") == "42"
    interp.shutdown()


def test_async_tool_awaited_inside_running_event_loop():
    """RLM.aforward() still calls execute() synchronously from inside the
    event loop; the coroutine must run on a helper loop, not deadlock."""
    import asyncio

    async def main() -> str:
        interp = MontyInterpreter(tools={"double": double_async})
        try:
            return interp.execute("double(4)")
        finally:
            interp.shutdown()

    assert asyncio.run(main()) == "8"


def test_async_tool_error_is_code_execution_error():
    from dspy.primitives.code_interpreter import CodeExecutionError

    async def boom() -> str:
        raise ValueError("async boom")

    interp = MontyInterpreter(tools={"boom": boom})
    with pytest.raises(CodeExecutionError, match="async boom"):
        interp.execute("boom()")
    interp.shutdown()


def test_async_tool_fires_tool_call_callbacks_with_awaited_result():
    import dspy
    from dspy.utils.callback import BaseCallback

    seen: list[tuple[str, object]] = []

    class Cb(BaseCallback):
        def on_interpreter_tool_call_end(self, call_id, outputs, exception=None):
            seen.append(("end", outputs))

    interp = MontyInterpreter(tools={"double": double_async}, callbacks=[Cb()])
    with dspy.context(callbacks=[]):
        interp.execute("double(5)")
    interp.shutdown()
    assert seen == [("end", 10)]
