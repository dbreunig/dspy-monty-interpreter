"""Host modules that sandbox code can import."""

import pytest
from dspy.primitives.code_interpreter import CodeExecutionError

from dspy_monty_interpreter import MontyInterpreter, MountDir


def test_match_statement():
    interp = MontyInterpreter()
    code = "match (1, 2):\n    case (a, b):\n        r = a + b\n    case _:\n        r = 0\nr"
    assert interp.execute(code) == "3"
    interp.shutdown()


@pytest.mark.parametrize("mod", ["csv", "tomllib", "hashlib", "bisect", "os.path"])
def test_monty_1_2_modules_import(mod):
    interp = MontyInterpreter()
    assert interp.execute(f"import {mod}\n'ok'") == "ok"
    interp.shutdown()


def test_instructions_list_monty_1_2_modules():
    text = MontyInterpreter.execution_instructions
    for mod in ("csv", "tomllib", "hashlib", "bisect", "os.path"):
        assert f" {mod}," in text or f" {mod}." in text


def test_os_walk_on_mount(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "x.txt").write_text("hi")
    interp = MontyInterpreter(
        mounts=MountDir(host_path=str(tmp_path), virtual_path="/data", mode="read-only")
    )
    out = interp.execute("import os\n[(d, f) for d, _, f in os.walk('/data')]")
    assert "('/data/sub', ['x.txt'])" in out
    interp.shutdown()


def _helpers_module():
    from dspy_monty_interpreter import ClassInstance, ExternalModule

    class Counter:
        def __init__(self):
            self.n = 0

        def bump(self, by: int = 1) -> int:
            self.n += by
            return self.n

    def add(a: int, b: int) -> int:
        return a + b

    def fail(msg: str) -> None:
        raise ValueError(msg)

    def counter():
        return ClassInstance(Counter(), allowed_methods={"bump"})

    return ExternalModule(
        {"add": add, "fail": fail, "counter": counter, "VERSION": "1", "_secret": 2,
         "text": ExternalModule({"upper": str.upper})}
    )


def test_external_module_functions_attrs_and_submodules():
    interp = MontyInterpreter(external_modules={"helpers": _helpers_module()})
    assert interp.execute("import helpers\nhelpers.add(2, 3)") == "5"
    assert interp.execute("from helpers import VERSION\nVERSION") == "1"
    assert interp.execute("import helpers.text\nhelpers.text.upper('a')") == "A"
    # State persists: the module stays imported in later steps.
    assert interp.execute("helpers.add(1, 1)") == "2"
    with pytest.raises(CodeExecutionError, match="no attribute '_secret'"):
        interp.execute("helpers._secret")
    interp.shutdown()


def test_external_module_host_error_becomes_code_execution_error():
    interp = MontyInterpreter(external_modules={"helpers": _helpers_module()})
    with pytest.raises(CodeExecutionError, match="ValueError: boom"):
        interp.execute("import helpers\nhelpers.fail('boom')")
    # The session survives a host-side exception.
    assert interp.execute("helpers.add(1, 2)") == "3"
    interp.shutdown()


def test_external_module_class_instance_keeps_host_state():
    interp = MontyInterpreter(external_modules={"helpers": _helpers_module()})
    interp.execute("import helpers\nc = helpers.counter()\nc.bump()")
    assert interp.execute("c.bump(5)") == "6"
    interp.shutdown()


def test_external_module_factory_is_lazy_and_called_once_per_session():
    calls = []

    def make():
        calls.append(1)
        return _helpers_module()

    factory = MontyInterpreter.factory(external_modules={"helpers": make})
    assert calls == []  # building the prompt does not call it
    interp = factory()
    interp.execute("1")
    assert calls == []  # nor does a feed that never imports it
    interp.execute("import helpers\nhelpers.add(1, 2)")
    interp.execute("import helpers\nhelpers.add(3, 4)")
    assert calls == [1]
    interp.shutdown()


def test_unknown_module_still_fails():
    interp = MontyInterpreter(external_modules={"helpers": _helpers_module()})
    with pytest.raises(CodeExecutionError, match="No module named 'numpy'"):
        interp.execute("import numpy")
    interp.shutdown()


def test_instructions_describe_external_modules():
    from dspy_monty_interpreter import ExternalModule

    text = MontyInterpreter.factory(
        external_modules={
            "helpers": _helpers_module(),
            "typed": ExternalModule({"f": len}, stubs="def f(x: str) -> int: ..."),
            "lazy": _helpers_module,
        }
    ).execution_instructions
    assert "Host modules you can import" in text
    assert "add(a: int, b: int) -> int" in text
    assert "VERSION" in text and "text (submodule)" in text and "_secret" not in text
    assert "```python\ndef f(x: str) -> int: ...\n```" in text
    assert "\n- lazy\n" in text or text.endswith("\n- lazy")
    assert text == text.strip()


def test_instructions_truncate_long_stubs():
    from dspy_monty_interpreter import ExternalModule

    stubs = "\n".join(f"def f{i}(x: int) -> int: ..." for i in range(1_000))
    text = MontyInterpreter.factory(
        external_modules={"big": ExternalModule({}, stubs=stubs)}
    ).execution_instructions
    assert "# ... (truncated)" in text
    assert len(text) < len(MontyInterpreter.execution_instructions) + 4_500


def test_instructions_omit_host_modules_section_without_modules():
    assert "Host modules" not in MontyInterpreter.factory().execution_instructions
    assert "Host modules" not in MontyInterpreter.factory(external_modules={}).execution_instructions


def test_rlm_prompt_includes_external_modules():
    import dspy

    factory = MontyInterpreter.factory(external_modules={"helpers": _helpers_module()})
    rlm = dspy.RLM("q -> a", interpreter_factory=factory)
    assert "add(a: int, b: int) -> int" in rlm.generate_action.signature.instructions
