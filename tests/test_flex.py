"""dspy.Flex modules running in Monty."""

import pydantic
import pytest

from dspy_monty_interpreter import MontyInterpreter, bridge

pytestmark = pytest.mark.skipif(not bridge.SUPPORTED, reason="dspy < 3.5 has no dspy.Flex")


def _flex(signature, src=None, **kwargs):
    import dspy

    flex = dspy.Flex(signature, interpreter_factory=MontyInterpreter, **kwargs)
    if src is not None:
        flex.load_state({"module_src": src})
    return flex


def test_baseline_string_signature():
    import dspy
    from dspy.utils.dummies import DummyLM

    with dspy.context(lm=DummyLM([{"answer": "4"}])):
        assert _flex("question -> answer")(question="2+2?").answer == "4"


def test_baseline_class_signature_keeps_descriptions_and_constraints():
    import dspy
    from dspy.utils.dummies import DummyLM

    class QA(dspy.Signature):
        """Answer with a count."""

        question: str = dspy.InputField(desc="what to count")
        count: int = dspy.OutputField(ge=0)

    lm = DummyLM([{"count": "3"}])
    with dspy.context(lm=lm):
        assert _flex(QA)(question="How many?").count == 3
    prompt = str(lm.history[0]["messages"])
    assert "what to count" in prompt and "Answer with a count." in prompt


def test_deterministic_module_with_helpers_and_nested_module():

    # Flex runs the first class with a forward, so the helper class comes second.
    src = '''
class SolveModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.doubler = Doubler()

    def forward(self, **inputs):
        import re
        n = int(re.findall(r"\\d+", inputs["text"])[0])
        doubled = self.doubler(n)
        assert "value" in doubled and doubled["value"] == doubled.value
        return dspy.Prediction(total=str(doubled.value))


class Doubler(dspy.Module):
    def forward(self, n):
        return dspy.Prediction(value=n * 2)
'''
    assert _flex("text -> total", src)(text="add 21").total == "42"


def test_predictor_results_support_subscripts_and_typed_outputs():
    import dspy
    from dspy.utils.dummies import DummyLM

    src = '''
class SolveModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.Predict("invoice: str -> qty: int, unit_price_cents: int")

    def forward(self, **inputs):
        e = self.extract(invoice=inputs["invoice"])
        return dspy.Prediction(total_cents=e["qty"] * e.unit_price_cents)
'''
    with dspy.context(lm=DummyLM([{"qty": "3", "unit_price_cents": "250"}])):
        assert _flex("invoice: str -> total_cents: int", src)(invoice="3 @ $2.50").total_cents == 750


def test_custom_types_cross_as_plain_data():
    import dspy

    class Item(pydantic.BaseModel):
        name: str
        price: float

    class Summary(pydantic.BaseModel):
        cheapest: str
        total: float

    class Shop(dspy.Signature):
        items: list[Item] = dspy.InputField()
        summary: Summary = dspy.OutputField()

    src = '''
class ShopModule(dspy.Module):
    def forward(self, items):
        cheapest = min(items, key=lambda item: item["price"])
        return dspy.Prediction(summary={"cheapest": cheapest["name"], "total": sum(i["price"] for i in items)})
'''
    items = [Item(name="a", price=2.0), Item(name="b", price=1.5)]
    summary = _flex(Shop, src)(items=items).summary
    assert summary == Summary(cheapest="b", total=3.5)


def test_flex_tools_are_callable_by_name():
    calls = []

    def lookup(city: str) -> str:
        """Population lookup."""
        calls.append(city)
        return "2161000"

    src = '''
class SolveModule(dspy.Module):
    def forward(self, **inputs):
        return dspy.Prediction(population=lookup(inputs["city"]))
'''
    assert _flex("city -> population", src, tools=[lookup])(city="Paris").population == "2161000"
    assert calls == ["Paris"]


def test_react_with_a_helper_defined_inside_forward():
    import dspy
    from dspy.utils.dummies import DummyLM

    src = '''
class SolveModule(dspy.Module):
    def __init__(self):
        super().__init__()

    def forward(self, **inputs):
        words = inputs["text"].split()

        def count(word: str) -> int:
            """Count occurrences of a word in the text."""
            return words.count(word)

        agent = dspy.ReAct("text, word -> n: int", tools=[count])
        return dspy.Prediction(n=agent(text=inputs["text"], word=inputs["word"]).n)
'''
    lm = DummyLM([
        {"next_thought": "count", "tool_calls": {"tool_calls": [{"name": "count", "args": {"word": "a"}}]}},
        {"next_thought": "done", "tool_calls": {"tool_calls": [{"name": "submit", "args": {"n": 2}}]}},
    ])
    with dspy.context(lm=lm):
        assert _flex("text, word -> n: int", src)(text="a b a", word="a").n == 2
    assert "Count occurrences of a word" in str(lm.history[0]["messages"])


def test_baseline_with_tools_runs_a_nested_rlm_on_monty():
    import dspy
    from dspy.utils.dummies import DummyLM

    def lookup(city: str) -> str:
        """Population lookup."""
        return "2161000"

    lm = DummyLM([
        {"plan": "look up", "code": "SUBMIT(population=lookup(city))"},
    ])
    with dspy.context(lm=lm):
        assert _flex("city -> population", tools=[lookup])(city="Paris").population == "2161000"
    assert "Monty" in str(lm.history[0]["messages"])


def test_predictor_budget_is_enforced():
    import dspy
    from dspy.primitives.code_interpreter import CodeInterpreterError
    from dspy.utils.dummies import DummyLM

    src = '''
class SolveModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.p = dspy.Predict("q -> a")

    def forward(self, **inputs):
        self.p(q="1")
        return dspy.Prediction(a=self.p(q="2").a)
'''
    with dspy.context(lm=DummyLM([{"a": "x"}] * 3)):
        with pytest.raises(CodeInterpreterError, match="budget"):
            _flex("q -> a", src, max_predictor_calls=1)(q="go")


def test_runtime_errors_report_the_module_line():
    from dspy.primitives.code_interpreter import CodeExecutionError

    src = '''
class SolveModule(dspy.Module):
    def __init__(self):
        super().__init__()

    def forward(self, **inputs):
        return dspy.Prediction(a=1 / 0)
'''
    with pytest.raises(CodeExecutionError, match="ZeroDivisionError"):
        _flex("q -> a", src)(q="go")
