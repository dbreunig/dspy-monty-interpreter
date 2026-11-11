"""REPL state carried across interpreters with SessionState."""

import pytest

from dspy_monty_interpreter import MontyInterpreter, SessionState, bridge

from _support import dspy_at_least, dummy_lm


def test_state_carries_variables_functions_and_classes():
    state = SessionState()
    first = MontyInterpreter(state=state)
    first.execute(
        "x = [1, 2]\ndef double(a):\n    return a * 2\nclass Box:\n    def __init__(self):\n        self.v = 3"
    )
    first.shutdown()
    assert state.data

    second = MontyInterpreter(state=state)
    try:
        assert second.execute("print(x, double(4), Box().v)") == "[1, 2] 8 3"
    finally:
        second.shutdown()


def test_empty_state_and_clear():
    state = SessionState()
    interp = MontyInterpreter(state=state)
    interp.shutdown()  # no session was started, so nothing is saved
    assert state.data is None
    assert repr(state) == "SessionState(empty)"

    interp = MontyInterpreter(state=state)
    interp.execute("y = 1")
    interp.shutdown()
    state.clear()
    fresh = MontyInterpreter(state=state)
    try:
        with pytest.raises(Exception, match="NameError"):
            fresh.execute("y")
    finally:
        fresh.shutdown()


def test_corrupt_state_starts_empty(caplog):
    interp = MontyInterpreter(state=SessionState(b"not a session"))
    try:
        assert interp.execute("print(1 + 1)") == "2"
    finally:
        interp.shutdown()
    assert "Could not restore" in caplog.text


@pytest.mark.skipif(not bridge.SUPPORTED, reason="dspy < 3.5 has no sandbox session")
def test_restored_session_still_has_the_dspy_namespace():
    import dspy
    from dspy.utils.dummies import DummyLM

    state = SessionState()
    first = MontyInterpreter(state=state)
    first.execute("import dspy\nn = 5")
    first.shutdown()

    second = MontyInterpreter(state=state)
    try:
        with dspy.context(lm=DummyLM([{"a": "ok"}])):
            assert second.execute("print(n, dspy.Predict('q -> a')(q='x').a)") == "5 ok"
    finally:
        second.shutdown()


@pytest.mark.skipif(not dspy_at_least(3, 5), reason="RLM takes history= from dspy 3.5")
def test_rlm_conversation_keeps_repl_variables():
    import dspy

    state = SessionState()
    factory = MontyInterpreter.factory(state=state)
    assert "persist" in factory.execution_instructions
    rlm = dspy.RLM("q -> a", interpreter_factory=factory, max_iters=2)

    with dspy.context(lm=dummy_lm([("remember", "total = 40\nSUBMIT(a='saved')")])):
        first = rlm(q="remember 40")
    with dspy.context(lm=dummy_lm([("recall", "SUBMIT(a=str(total + 2))")])):
        second = rlm(q="add 2", history=first.history)
    assert second.a == "42"
