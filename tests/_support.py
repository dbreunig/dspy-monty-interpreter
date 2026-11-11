"""Helpers shared by the test modules."""

import re


def dspy_at_least(major: int, minor: int) -> bool:
    import dspy

    installed = tuple(int(part) for part in re.findall(r"\d+", dspy.__version__)[:2])
    return installed >= (major, minor)


def plan_field() -> str:
    """RLM's plan output field: dspy 3.5 renamed it from ``reasoning`` for
    LMs without native reasoning."""
    return "plan" if dspy_at_least(3, 5) else "reasoning"


def dummy_lm(steps: list[tuple[str, str]]):
    from dspy.utils.dummies import DummyLM

    return DummyLM([{plan_field(): r, "code": c} for r, c in steps])


async def double_async(x: int) -> int:
    """Async tool stub."""
    import asyncio

    await asyncio.sleep(0)
    return x * 2
