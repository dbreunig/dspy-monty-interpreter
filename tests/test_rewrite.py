"""Source rewrites that let dspy code run in Monty."""

import ast

import pytest

from dspy_monty_interpreter import MontyInterpreter, bridge, rewrite

needs_bridge = pytest.mark.skipif(not bridge.SUPPORTED, reason="dspy < 3.5 has no sandbox session")


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("import dspy\n", "pass\n"),
        ("import dspy as d\n", "d = dspy\n"),
        ("import os, dspy\n", "import os\n"),
        ("from dspy import Predict, Signature as S\n", "Predict = dspy.Predict; S = dspy.Signature\n"),
        ("from dspy.primitives import x\n", "from dspy.primitives import x\n"),
        ("import json\n", "import json\n"),
    ],
)
def test_dspy_imports(code, expected):
    assert rewrite.rewrite(code) == expected


@pytest.mark.parametrize(
    "code",
    [
        "class M(dspy.Module):\n    def forward(self):\n        return 1\n\n\nx = 1\n",
        "class M(\n    dspy.Module,\n):\n    def __init__(self):\n        super().__init__()\n\n\ny = 2\n",
        "class M(dspy.Module): pass\n",
        "from dspy import (\n    Predict,\n    ChainOfThought,\n)\nz = 3\n",
        'class S(dspy.Signature):\n    """Doc."""\n\n    q: str = dspy.InputField(desc="x")\n'
        "    a: int = dspy.OutputField()\n\n\nw = 4\n",
    ],
)
def test_rewrites_keep_line_numbers_and_parse(code):
    out = rewrite.rewrite(code)
    ast.parse(out)
    assert out.count("\n") == code.count("\n")
    if code.count("\n") > 1:  # code after the rewritten statement stays on its line
        last = code.rstrip("\n").splitlines()[-1]
        assert out.rstrip("\n").splitlines()[-1] == last


def test_module_subclass_loses_base_and_gains_call():
    out = rewrite.rewrite("class M(dspy.Module, Base):\n    def __init__(self):\n        super().__init__()\n")
    tree = ast.parse(out)
    cls = tree.body[0]
    assert [ast.unparse(b) for b in cls.bases] == ["Base"]
    assert [f.name for f in cls.body] == ["__init__", "__call__"]
    assert "super()" not in out


def test_signature_class_becomes_a_host_signature_call():
    code = (
        "class QA(dspy.Signature):\n"
        '    """Answer briefly."""\n'
        "    question: list[str] = dspy.InputField(desc='q')\n"
        "    answer: int = dspy.OutputField(ge=0)\n"
    )
    out = rewrite.rewrite(code)
    assert out.splitlines()[0] == (
        "QA = dspy.Signature('question: list[str] -> answer: int', 'Answer briefly.', "
        "{'question': dict(desc='q'), 'answer': dict(ge=0)})"
    )


def test_signature_class_with_methods_is_left_alone():
    code = "class QA(dspy.Signature):\n    def helper(self):\n        return 1\n"
    assert rewrite.rewrite(code) == code


def test_functions_records_signature_and_docstring():
    info = rewrite.functions('def f(a: int, b: str = "x", *, c=None):\n    """Doc."""\n    return a\n')["f"]
    assert info.doc == "Doc."
    assert info.params == [
        {"name": "a", "annotation": "int"},
        {"name": "b", "annotation": "str", "default": "x"},
        {"name": "c", "default": None},
    ]


@needs_bridge
def test_repl_code_can_define_modules_and_signatures():
    import dspy
    from dspy.utils.dummies import DummyLM

    interp = MontyInterpreter()
    try:
        interp.execute(
            "import dspy\n"
            "class Topic(dspy.Signature):\n"
            '    """Name the topic."""\n'
            "    text: str = dspy.InputField()\n"
            "    topic: str = dspy.OutputField()\n"
            "class Pipeline(dspy.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.p = dspy.Predict(Topic)\n"
            "    def forward(self, text):\n"
            "        return self.p(text=text)\n"
        )
        with dspy.context(lm=DummyLM([{"topic": "greetings"}])):
            assert interp.execute("r = Pipeline()(text='hi')\nprint(r['topic'], r)") == (
                "greetings Prediction(topic='greetings')"
            )
    finally:
        interp.shutdown()


@needs_bridge
def test_tracebacks_point_at_the_written_line():
    from dspy.primitives.code_interpreter import CodeExecutionError

    interp = MontyInterpreter()
    try:
        code = "import dspy\nclass M(dspy.Module):\n    def forward(self):\n        return 1\n\n\nx = 1 / 0\n"
        with pytest.raises(CodeExecutionError) as e:
            interp.execute(code)
        assert "ZeroDivisionError" in str(e.value)
        assert "line 7" in str(e.value.__cause__.display("traceback"))
    finally:
        interp.shutdown()


def test_functions_skip_methods_and_prefer_module_level():
    code = (
        "def search(query: str, k: int = 3):\n    pass\n"
        "class C:\n    def search(self, q):\n        pass\n"
        "    def forward(self):\n        def helper(x: int):\n            pass\n"
    )
    found = rewrite.functions(code)
    assert [p["name"] for p in found["search"].params] == ["query", "k"]
    assert "helper" in found and "forward" not in found


def test_unbuildable_default_is_an_execution_error():
    from dspy.primitives.code_interpreter import CodeExecutionError

    interp = MontyInterpreter()
    try:
        with pytest.raises(CodeExecutionError):
            interp.execute("def f(x={[1]: 2}):\n    return x\nf()")
    finally:
        interp.shutdown()
