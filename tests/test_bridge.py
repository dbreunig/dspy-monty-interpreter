"""The dspy 3.5 sandbox session bridge: object proxies and sub-agents."""

import pytest
from dspy.primitives.code_interpreter import CodeExecutionError

from dspy_monty_interpreter import MontyInterpreter, bridge

from _support import dspy_at_least, dummy_lm

needs_bridge = pytest.mark.skipif(not bridge.SUPPORTED, reason="dspy < 3.5 has no sandbox session")


class _Report:
    def __init__(self, title, pages):
        self.title = title
        self.pages = pages
        self._secret = "hidden"
        self.callback = print

    def summary(self):
        return f"{self.title} ({self.pages}p)"

    def _private(self):
        return "hidden"


def test_bridge_supported_on_dspy_35_and_later():
    """Fails when a dspy release moves the internals the bridge relies on."""
    assert bridge.SUPPORTED == dspy_at_least(3, 5)


def test_raw_variables_do_not_persist():
    interp = MontyInterpreter()
    try:
        interp.execute("doc = _raw_doc.upper()", variables={"_raw_doc": "hello"})
        assert interp.execute("doc") == "HELLO"
        with pytest.raises(CodeExecutionError, match="NameError"):
            interp.execute("_raw_doc")
    finally:
        interp.shutdown()


def test_object_inputs_cross_as_proxies():
    interp = MontyInterpreter()
    try:
        report = _Report("Q3", 12)
        assert interp.execute("r.title", variables={"r": report}) == "Q3"
        assert interp.execute("[x.pages for x in rs]", variables={"rs": [report]}) == "[12]"
        assert interp.execute("SUBMIT(a=r)", variables={"r": report}).output == {"a": report}
    finally:
        interp.shutdown()


@pytest.mark.parametrize(
    ("code", "error"),
    [
        ("r._secret", "AttributeError"),
        ("r._private()", "AttributeError"),
        ("r.callback", "AttributeError"),
        ("r.summary()", "AttributeError"),
        ("r.__class__('x', 1)", "AttributeError"),
        ("type(r)('x', 1)", "TypeError"),
    ],
)
def test_object_proxies_hide_private_names_and_stored_callables(code, error):
    interp = MontyInterpreter()
    try:
        with pytest.raises(CodeExecutionError, match=error):
            interp.execute(code, variables={"r": _Report("Q3", 12)})
    finally:
        interp.shutdown()


@needs_bridge
def test_rlm_bridge_installs_without_warning(caplog):
    import dspy

    lm = dummy_lm([("go", "SUBMIT(a='ok')")])
    with dspy.context(lm=lm):
        rlm = dspy.RLM("q -> a", interpreter_factory=MontyInterpreter, max_iters=2)
        assert rlm(q="x").a == "ok"
    assert "sub-agents are unavailable" not in caplog.text
    assert bridge.SUB_AGENT_INSTRUCTIONS in rlm.generate_action.signature.instructions


@needs_bridge
def test_rlm_object_input_reaches_sandbox():
    import dspy

    lm = dummy_lm([("go", "SUBMIT(a=f'{report.title} ({report.pages}p)')")])
    with dspy.context(lm=lm):
        rlm = dspy.RLM("report -> a", interpreter_factory=MontyInterpreter, max_iters=2)
        assert rlm(report=_Report("Q3", 12)).a == "Q3 (12p)"


@needs_bridge
def test_rlm_predict_sub_agent_runs_on_host():
    import dspy
    from dspy.utils.dummies import DummyLM

    lm = DummyLM([
        {"plan": "ask", "code": "import dspy\nr = dspy.Predict('text -> topic')(text='hi there')\nprint(r.topic)"},
        {"topic": "greetings"},
        {"plan": "done", "code": "SUBMIT(a=r.topic)"},
    ])
    with dspy.context(lm=lm):
        rlm = dspy.RLM("q -> a", interpreter_factory=MontyInterpreter, max_iters=3)
        result = rlm(q="x")
    assert result.a == "greetings"
    assert result.trajectory[0]["output"] == "greetings"


@needs_bridge
def test_rlm_nested_rlm_sub_agent_gets_its_own_sandbox():
    import dspy
    from dspy.utils.dummies import DummyLM

    lm = DummyLM([
        {
            "plan": "p",
            "code": "import dspy\nsecret = 'outer'\nr = dspy.RLM('q -> a', max_iters=2)(q='inner')\nprint(r.a)",
        },
        {"plan": "inner", "code": "try:\n    seen = secret\nexcept NameError:\n    seen = 'isolated'\nSUBMIT(a=seen)"},
        {"plan": "done", "code": "SUBMIT(a=r.a)"},
    ])
    with dspy.context(lm=lm):
        rlm = dspy.RLM("q -> a", interpreter_factory=MontyInterpreter, max_iters=3)
        assert rlm(q="outer").a == "isolated"


@needs_bridge
def test_rlm_react_sub_agent_uses_provided_tools():
    import dspy
    from dspy.utils.dummies import DummyLM

    calls = []

    def lookup(city: str) -> str:
        """Population lookup."""
        calls.append(city)
        return "2161000"

    lm = DummyLM([
        {
            "plan": "p",
            "code": (
                "import dspy\nr = dspy.ReAct('city -> population', tools=[lookup])(city='Paris')\nprint(r.population)"
            ),
        },
        {"next_thought": "look it up", "tool_calls": {"tool_calls": [{"name": "lookup", "args": {"city": "Paris"}}]}},
        {"next_thought": "done", "tool_calls": {"tool_calls": [{"name": "submit", "args": {"population": "2161000"}}]}},
        {"plan": "done", "code": "SUBMIT(a=r.population)"},
    ])
    with dspy.context(lm=lm):
        rlm = dspy.RLM("q -> a", tools=[lookup], interpreter_factory=MontyInterpreter, max_iters=3)
        assert rlm(q="x").a == "2161000"
    assert calls == ["Paris"]


@needs_bridge
def test_rlm_react_rejects_repl_functions():
    import dspy

    lm = dummy_lm([
        ("p", "import dspy\ndef f(x):\n    return x\ndspy.ReAct('q -> a', tools=[f])"),
        ("done", "SUBMIT(a='stopped')"),
    ])
    with dspy.context(lm=lm):
        rlm = dspy.RLM("q -> a", interpreter_factory=MontyInterpreter, max_iters=3)
        result = rlm(q="x")
    assert "ReAct tools must be the provided tools" in result.trajectory[0]["output"]


def test_caller_wrapped_objects_keep_their_own_policy():
    from dspy_monty_interpreter import ClassInstance

    interp = MontyInterpreter()
    try:
        report = ClassInstance(_Report("Q3", 12), allowed_methods={"summary"})
        assert interp.execute("r.summary()", variables={"r": report}) == "Q3 (12p)"
    finally:
        interp.shutdown()


def test_proxy_instructions_on_every_dspy_version():
    assert bridge.PROXY_INSTRUCTIONS in MontyInterpreter.execution_instructions
    assert (bridge.SUB_AGENT_INSTRUCTIONS in MontyInterpreter.execution_instructions) == bridge.SUPPORTED


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("doc = _raw_doc.upper()", "doc = _raw_doc().upper()"),
        ("x = obj._raw_doc + _raw_doc", "x = obj._raw_doc + _raw_doc()"),
        ("s = '_raw_doc'  # _raw_doc\n", "s = '_raw_doc'  # _raw_doc\n"),
        ("a, b = _raw_doc, _raw_doc", "a, b = _raw_doc(), _raw_doc()"),
        ("_raw_document = _raw_doc", "_raw_document = _raw_doc()"),
    ],
)
def test_raw_rewrite_only_touches_standalone_names(code, expected):
    rewritten, lookups = bridge.take_raw_variables(code, {"_raw_doc": "x"})
    assert rewritten == expected
    assert lookups["_raw_doc"]() == "x"


@needs_bridge
@pytest.mark.parametrize("code", ["p.save('out.json')", "p.load('out.json')", "p.signature", "p.forward(q='x')"])
def test_sub_agents_expose_only_their_call(code, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    interp = MontyInterpreter()
    try:
        interp.execute("import dspy\np = dspy.Predict('q -> a')")
        with pytest.raises(CodeExecutionError, match="AttributeError"):
            interp.execute(code)
        assert not (tmp_path / "out.json").exists()
    finally:
        interp.shutdown()


@needs_bridge
def test_rlm_sub_agent_calls_count_against_max_llm_calls():
    import dspy
    from dspy.utils.dummies import DummyLM

    lm = DummyLM([
        {"plan": "p", "code": "import dspy\np = dspy.Predict('q -> a')\nfirst = p(q='1').a\np(q='2')"},
        {"a": "one"},
        {"plan": "done", "code": "SUBMIT(a=first)"},
    ])
    with dspy.context(lm=lm):
        rlm = dspy.RLM("q -> a", interpreter_factory=MontyInterpreter, max_iters=2, max_llm_calls=1)
        result = rlm(q="x")
    assert "LLM call limit exceeded" in result.trajectory[0]["output"]
    assert result.a == "one"
