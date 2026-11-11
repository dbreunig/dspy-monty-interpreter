"""Opt-in type checking of each snippet before it runs."""

from typing import Any

import pydantic
import pytest
from dspy.primitives.code_interpreter import CodeExecutionError

from dspy_monty_interpreter import MontyInterpreter, bridge, typecheck

from _support import dummy_lm


class _Doc(pydantic.BaseModel):
    text: str


def search(query: str, k: int = 5) -> list[str]:
    return [query] * k


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        (str, "str"),
        (list[str], "list[str]"),
        (str | None, "str | None"),
        (dict[str, int], "dict[str, int]"),
        (_Doc, "Any"),
        (list[_Doc], "Any"),
        (Any, "Any"),
    ],
)
def test_annotation_text(annotation, expected):
    assert typecheck.annotation_text(annotation) == expected


def test_function_stub_keeps_parameters():
    assert typecheck.function_stub("search", search) == "def search(query: str, k: int = ...) -> list[str]: ..."

    def lookup(doc: _Doc, *args, flag: bool = False, **kwargs) -> _Doc:
        return doc

    assert typecheck.function_stub("lookup", lookup) == (
        "def lookup(doc: Any, *args: Any, flag: bool = ..., **kwargs: Any) -> Any: ..."
    )


def test_rejected_snippet_does_not_run():
    interp = MontyInterpreter(tools={"search": search}, type_check=True)
    try:
        interp.execute("n = 1")
        with pytest.raises(CodeExecutionError, match=r"(?s)Type check failed.*invalid-argument-type"):
            interp.execute("n = 2\nsearch(query=3)")
        with pytest.raises(CodeExecutionError, match="unresolved-reference"):
            interp.execute("n = 3\nprint(undefined_name)")
        assert interp.execute("print(n)") == "1"
        assert interp.execute("print(len(search('a', 2)))") == "2"
    finally:
        interp.shutdown()


def test_variables_can_be_reassigned_across_snippets():
    interp = MontyInterpreter(type_check=True)
    try:
        interp.execute("n = 1\nresult = None\nfor i in range(2):\n    last = i")
        assert interp.execute("n = n + 0.5\nresult = 'done'\nlast = str(last)\nprint(n, result, last)") == (
            "1.5 done 1"
        )
    finally:
        interp.shutdown()


def test_assigned_names():
    code = "a = 1\nb, (c, d) = 1, (2, 3)\nfor e in x:\n    f += 1\ng: int = 1\ndef h():\n    i = 1\n[j for j in x]"
    assert typecheck.assigned_names(code) == {"a", "b", "c", "d", "e", "f"}


def test_input_variables_are_declared_by_type():
    interp = MontyInterpreter(type_check=True)
    try:
        assert interp.execute("print(context.upper())", variables={"context": "hi"}) == "HI"
        assert interp.execute("print(len(rows))", variables={"rows": [1, 2]}) == "2"
        with pytest.raises(CodeExecutionError, match="unsupported-operator"):
            interp.execute("context + 1", variables={"context": "hi"})
    finally:
        interp.shutdown()


@pytest.mark.skipif(not bridge.SUPPORTED, reason="dspy < 3.5 has no sandbox session")
def test_dspy_namespace_passes_the_checker():
    interp = MontyInterpreter(type_check=True)
    try:
        interp.execute(
            "import dspy\n"
            "class M(dspy.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "    def forward(self, a):\n"
            "        return dspy.Prediction(b=a)\n"
        )
        assert interp.execute("r = M()(a=1)\nprint(r.b, r['b'], 'b' in r)") == "1 1 True"
    finally:
        interp.shutdown()


def test_instructions_mention_type_checking():
    assert "type-checked" in MontyInterpreter(type_check=True).execution_instructions
    assert "type-checked" in MontyInterpreter.factory(type_check=True).execution_instructions
    assert "type-checked" not in MontyInterpreter.execution_instructions


def test_rlm_sees_the_type_error_and_recovers():
    import dspy

    lm = dummy_lm([
        ("try", "hits = search(query=3)"),
        ("fix", "hits = search(query='3', k=1)\nprint(hits)"),
        ("done", "SUBMIT(a=hits[0])"),
    ])
    with dspy.context(lm=lm):
        rlm = dspy.RLM(
            "q -> a", tools=[search], interpreter_factory=MontyInterpreter.factory(type_check=True), max_iters=3
        )
        result = rlm(q="x")
    assert "Type check failed" in result.trajectory[0]["output"]
    assert result.a == "3"


@pytest.mark.skipif(not bridge.SUPPORTED, reason="dspy < 3.5 has no dspy.Flex")
def test_flex_runs_with_type_checking():
    import dspy

    def lookup(city: str) -> str:
        """Population lookup."""
        return "2161000"

    src = '''
class SolveModule(dspy.Module):
    def forward(self, **inputs):
        return dspy.Prediction(population=lookup(inputs["city"]))
'''
    factory = MontyInterpreter.factory(type_check=True)
    flex = dspy.Flex("city -> population", tools=[lookup], interpreter_factory=factory)
    flex.load_state({"module_src": src})
    assert flex(city="Paris").population == "2161000"


def test_literal_annotations_are_enforced():
    from typing import Literal

    def pick(mode: Literal["a", "b"]) -> str:
        return mode

    interp = MontyInterpreter(tools={"pick": pick}, type_check=True)
    try:
        assert interp.execute("print(pick('a'))") == "a"
        with pytest.raises(CodeExecutionError, match="invalid-argument-type"):
            interp.execute("pick('c')")
    finally:
        interp.shutdown()
