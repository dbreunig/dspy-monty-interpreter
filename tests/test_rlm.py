"""MontyInterpreter driving dspy.RLM with a scripted LM."""

import pytest
from dspy.primitives.code_interpreter import CodeInterpreterError

from dspy_monty_interpreter import MontyInterpreter, MountDir

from _support import double_async, dummy_lm


def test_rlm_accepts_class_as_interpreter_factory():
    import dspy

    assert isinstance(MontyInterpreter(), dspy.CodeInterpreter)
    rlm = dspy.RLM("q -> a", interpreter_factory=MontyInterpreter)
    assert "(yield) are NOT supported" in rlm.generate_action.signature.instructions


def test_rlm_forward_runs_tool_and_typed_submit_through_monty():
    """Full RLM loop on dspy >= 3.4 with a scripted LM: tool call, print
    output fed back, typed SUBMIT parsed into the output field."""
    import dspy

    calls: list[str] = []

    def lookup(city: str) -> str:
        """Population lookup."""
        calls.append(city)
        return "2161000"

    lm = dummy_lm([("look", "pop = lookup('Paris')\nprint(pop)"), ("done", "SUBMIT(population=int(pop))")])
    with dspy.context(lm=lm):
        rlm = dspy.RLM("city -> population: int", tools=[lookup], interpreter_factory=MontyInterpreter, max_iters=3)
        result = rlm(city="Paris")
    assert result.population == 2161000
    assert calls == ["Paris"]
    assert result.trajectory[0]["output"] == "2161000"


def test_rlm_call_time_interpreter_factory_override(tmp_path):
    """dspy >= 3.4: interpreter_factory= at call time overrides the module's
    factory for that call only, prompt instructions included."""
    import dspy

    (tmp_path / "note.txt").write_text("from mount")
    mounted = MontyInterpreter.factory(
        mounts=MountDir(virtual_path="/data", host_path=str(tmp_path), mode="read-only")
    )
    lm = dummy_lm([("read", "SUBMIT(a=open('/data/note.txt').read())")])
    with dspy.context(lm=lm):
        rlm = dspy.RLM("q -> a", interpreter_factory=MontyInterpreter, max_iters=2)
        assert rlm(q="x", interpreter_factory=mounted).a == "from mount"


def test_rlm_uses_configured_interpreter_factory_and_its_instructions():
    """dspy.configure/context(interpreter_factory=...) selects Monty for a
    plain dspy.RLM() and RLM picks up Monty's execution_instructions."""
    import dspy

    lm = dummy_lm([("go", "SUBMIT(a='hi')")])
    with dspy.context(lm=lm, interpreter_factory=MontyInterpreter.factory(request_timeout=5.0)):
        rlm = dspy.RLM("q -> a", max_iters=2)
        assert "(yield) are NOT supported" in rlm.generate_action.signature.instructions
        assert rlm(q="x").a == "hi"


def test_rlm_acall_awaits_async_tool_through_monty():
    import asyncio

    import dspy

    lm = dummy_lm([("call", "y = double(2)\nprint(y)"), ("done", "SUBMIT(a=str(y))")])
    double = dspy.Tool(double_async, name="double")

    async def main():
        with dspy.context(lm=lm):
            rlm = dspy.RLM("q -> a", tools=[double], interpreter_factory=MontyInterpreter, max_iters=3)
            return await rlm.acall(q="x")

    result = asyncio.run(main())
    assert result.a == "4"
    assert result.trajectory[0]["output"] == "4"


def test_rlm_sandbox_serializable_inputs_are_injected():
    """dspy >= 3.4 injects SandboxSerializable inputs via start() + execute()
    with text or base64 payloads before the first iteration."""
    import dspy
    from dspy.primitives.sandbox_serializable import SandboxSerializable

    class Doc(SandboxSerializable):
        def __init__(self, text):
            self.text = text

        def sandbox_setup(self):
            return "import json"

        def to_sandbox(self):
            return self.text.encode()

        def sandbox_assignment(self, name, raw):
            return f"{name} = {raw}.upper()"

        def rlm_preview(self, max_chars=500):
            return "a doc"

    class Blob(SandboxSerializable):
        def sandbox_setup(self):
            return ""

        def to_sandbox(self):
            return bytes([0xFF, 0x00, 0x41])  # not valid UTF-8 -> base64 path

        def sandbox_assignment(self, name, raw):
            return f"{name} = len({raw})"

        def rlm_preview(self, max_chars=500):
            return "blob"

    lm = dummy_lm([("go", "SUBMIT(a=doc + str(blob))")])
    with dspy.context(lm=lm):
        rlm = dspy.RLM("doc, blob -> a", interpreter_factory=MontyInterpreter, max_iters=2)
        assert rlm(doc=Doc("hello"), blob=Blob()).a == "HELLO3"


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
