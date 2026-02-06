"""Tests for MontyInterpreter."""

import pytest
from dspy.primitives.code_interpreter import CodeInterpreter, CodeInterpreterError, FinalOutput

from dspy_monty_interpreter import MontyInterpreter


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


# --- Tool call caching during replay ---


def test_tool_called_once_across_accumulations():
    call_count = [0]

    def counted_tool(x: str) -> str:
        call_count[0] += 1
        return f"result_{call_count[0]}"

    interp = MontyInterpreter(tools={"counted_tool": counted_tool})
    interp.execute('a = counted_tool(x="first")')
    assert call_count[0] == 1

    # Second execute replays the first tool call from cache
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
    # "first" should NOT be re-called (cached), only "second" is new
    assert call_log == ["first", "second"]

    result = interp.execute("print(a + ' ' + b)")
    assert result == "got_first got_second"
    # Still only 2 total live calls
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
    interp = MontyInterpreter()
    interp.execute("x = 10")

    with pytest.raises(CodeInterpreterError):
        interp.execute("undefined_var")

    # x should still be accessible — failed code was not accumulated
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
