"""Async and remote Monty pools behind the sync interpreter interface.

The tests drive the async machinery with ``AsyncMonty``, the local pool
that hands out the same ``AsyncMontySession`` as ``AsyncMontyRemote``.
"""

import threading

import pytest
from dspy.primitives.code_interpreter import CodeExecutionError, CodeInterpreterError
from pydantic_monty import AsyncMonty

from dspy_monty_interpreter import MontyInterpreter, SessionState, bridge

from _support import double_async


def _interp(**kwargs):
    return MontyInterpreter(remote=AsyncMonty, **kwargs)


def test_state_persists_and_errors_are_recoverable():
    interp = _interp()
    try:
        interp.execute("x = 41")
        with pytest.raises(CodeExecutionError, match="ZeroDivisionError"):
            interp.execute("1 / 0")
        assert interp.execute("print(x + 1)") == "42"
    finally:
        interp.shutdown()


def test_sync_and_async_tools():
    interp = _interp(tools={"add": lambda a, b: a + b, "double": double_async})
    try:
        assert interp.execute("print(add(1, b=2), double(x=4))") == "3 8"
    finally:
        interp.shutdown()


def test_submit_and_type_check():
    interp = _interp(type_check=True, output_fields=[{"name": "a"}])
    try:
        with pytest.raises(CodeExecutionError, match="Type check failed"):
            interp.execute("1 + 'a'")
        assert interp.execute("SUBMIT(7)").output == {"a": 7}
    finally:
        interp.shutdown()


def test_session_state_round_trips():
    state = SessionState()
    first = _interp(state=state)
    first.execute("def f(n):\n    return n * 3")
    first.shutdown()
    second = _interp(state=state)
    try:
        assert second.execute("print(f(3))") == "9"
    finally:
        second.shutdown()


def test_shutdown_stops_the_loop_thread():
    before = {t for t in threading.enumerate() if t.name == "monty-async-pool"}
    interp = _interp()
    interp.execute("x = 1")
    interp.shutdown()
    after = {t for t in threading.enumerate() if t.name == "monty-async-pool"}
    assert after <= before


@pytest.mark.skipif(not bridge.SUPPORTED, reason="dspy < 3.5 has no sandbox session")
def test_rlm_sub_agents_see_the_callers_dspy_context():
    """Sub-agent LM calls run on the host through the event loop thread and
    must still use the LM and call budget from the caller's dspy.context."""
    import dspy
    from dspy.utils.dummies import DummyLM

    lm = DummyLM([
        {"plan": "ask", "code": "r = dspy.Predict('text -> topic')(text='hi')\nprint(r['topic'])"},
        {"topic": "greetings"},
        {"plan": "done", "code": "SUBMIT(a=r.topic)"},
    ])
    with dspy.context(lm=lm):
        rlm = dspy.RLM("q -> a", interpreter_factory=lambda: _interp(), max_iters=3)
        result = rlm(q="x")
    assert result.a == "greetings"
    assert result.trajectory[0]["output"] == "greetings"


def test_tools_see_the_callers_context_variables():
    import contextvars

    marker = contextvars.ContextVar("marker", default="unset")

    interp = _interp(tools={"read": lambda: marker.get()})
    try:
        token = marker.set("caller")
        try:
            assert interp.execute("print(read())") == "caller"
        finally:
            marker.reset(token)
    finally:
        interp.shutdown()


def test_unreachable_remote_raises():
    interp = MontyInterpreter(remote="http://127.0.0.1:9", request_timeout=2.0)
    try:
        with pytest.raises(CodeInterpreterError, match="Could not open a Monty session"):
            interp.execute("1 + 1")
    finally:
        interp.shutdown()
