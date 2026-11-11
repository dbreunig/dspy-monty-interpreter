"""Running code: output, state, SUBMIT, code fences, and errors."""

import pytest
from dspy.primitives.code_interpreter import (
    CodeInterpreter,
    CodeInterpreterError,
    FinalOutput,
)

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


def test_unknown_resource_limit_key_is_rejected():
    interp = MontyInterpreter(resource_limits={"bogus": 1})
    with pytest.raises(ValueError, match="bogus"):
        interp.execute("1")


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
