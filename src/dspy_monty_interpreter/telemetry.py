"""OpenTelemetry for the interpreter and the Monty sandbox.

``instrument()`` sends Monty's own telemetry to OpenTelemetry: a span per
session, with a child per snippet run and a grandchild per host call, plus
metrics and logs. Independently, every ``execute()`` gets a
``monty.execute`` span recording how the call ended, when
``opentelemetry-api`` is installed. Without it both are inert.
"""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable
from typing import Any

from dspy.primitives.code_interpreter import FinalOutput

try:
    from opentelemetry import trace as _trace
except ImportError:  # opentelemetry-api is optional
    _trace = None

_SCOPE = "dspy_monty_interpreter"
_lock = threading.Lock()


def instrument(
    *,
    tracer_provider: Any | None = None,
    meter_provider: Any | None = None,
    logger_provider: Any | None = None,
) -> bool:
    """Send Monty's spans and metrics, and logs when ``logger_provider`` is
    given, to OpenTelemetry. Spans and metrics use the global providers
    unless others are given.

    Monty installs telemetry once per process. Returns False when it was
    already installed, here or by a direct ``pydantic_monty.instrument_telemetry`` call.
    """
    from opentelemetry import metrics, trace
    from pydantic_monty import instrument_telemetry

    tracer = (tracer_provider or trace.get_tracer_provider()).get_tracer("pydantic_monty")
    meter = (meter_provider or metrics.get_meter_provider()).get_meter("pydantic_monty")
    logger = logger_provider.get_logger("pydantic_monty") if logger_provider is not None else None
    with _lock:
        try:
            instrument_telemetry(tracer=tracer, meter=meter, logger=logger)
        except RuntimeError as e:
            if "already configured" in str(e):
                return False
            raise
    return True


def traced_execute(execute: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``MontyInterpreter.execute`` in a ``monty.execute`` span that
    records the snippet's size and how it ended."""
    if _trace is None:
        return execute

    @functools.wraps(execute)
    def wrapper(self: Any, code: str, variables: dict[str, Any] | None = None) -> Any:
        tracer = _trace.get_tracer(_SCOPE)
        with tracer.start_as_current_span("monty.execute") as span:
            span.set_attribute("monty.code.lines", code.count("\n") + 1)
            try:
                result = execute(self, code, variables)
            except BaseException as e:
                span.set_attribute("monty.outcome", "error")
                span.set_attribute("monty.error.type", type(e).__name__)
                raise
            if isinstance(result, FinalOutput):
                outcome = "submit"
            else:
                outcome = "none" if result is None else "output"
            span.set_attribute("monty.outcome", outcome)
            return result

    return wrapper
