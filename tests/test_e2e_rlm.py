"""End-to-end test: MontyInterpreter driving a real dspy.RLM against a live LLM.

Skipped unless an API key is present in .env (or the host environment).
Run explicitly with:  pytest -m e2e
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is a dev dep
    load_dotenv = None

import dspy

from dspy_monty_interpreter import MontyInterpreter


_PROJECT_ROOT = Path(__file__).resolve().parent.parent

if load_dotenv is not None:
    load_dotenv(_PROJECT_ROOT / ".env", override=False)


def _select_lm() -> dspy.LM | None:
    """Pick an LM based on available API keys. Prefer Anthropic."""
    override = os.getenv("DSPY_LM_MODEL")
    if os.getenv("ANTHROPIC_API_KEY"):
        return dspy.LM(override or "anthropic/claude-sonnet-4-6", max_tokens=30_720)
    if os.getenv("OPENAI_API_KEY"):
        return dspy.LM(override or "openai/gpt-5.4-mini-2026-03-17", max_tokens=30_720)
    return None


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        _select_lm() is None,
        reason="No ANTHROPIC_API_KEY or OPENAI_API_KEY found in environment or .env",
    ),
]


def test_rlm_computes_via_monty():
    """RLM should pick Monty, write Python, and return the right answer."""
    dspy.configure(lm=_select_lm())

    interpreter = MontyInterpreter()
    rlm = dspy.RLM(
        "numbers: list[int] -> product: int",
        interpreter=interpreter,
        max_iterations=5,
        max_llm_calls=3,
    )

    result = rlm(numbers=[2, 3, 5, 7])

    # 2 * 3 * 5 * 7 = 210 — simple enough that a small model reliably solves it
    # via code, which exercises the full Monty execution path.
    assert int(result.product) == 210


def test_rlm_with_custom_tool():
    """RLM can invoke a user-provided tool through Monty's external_functions."""
    dspy.configure(lm=_select_lm())

    call_log: list[str] = []

    def lookup_city_population(city: str) -> str:
        """Return population of a city. Tiny stub database."""
        call_log.append(city)
        return {
            "tokyo": "13960000",
            "paris": "2161000",
            "lagos": "15388000",
        }.get(city.lower(), "0")

    interpreter = MontyInterpreter(tools={"lookup_city_population": lookup_city_population})
    rlm = dspy.RLM(
        "city: str -> population: int",
        interpreter=interpreter,
        tools=[lookup_city_population],
        max_iterations=5,
        max_llm_calls=3,
    )

    result = rlm(city="Paris")

    assert int(result.population) == 2161000
    assert any(c.lower() == "paris" for c in call_log), (
        f"tool should have been called for Paris, got calls={call_log}"
    )
