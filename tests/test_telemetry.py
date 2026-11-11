"""OpenTelemetry spans for execute() and Monty's own telemetry.

Monty installs telemetry once per process, so the end-to-end check runs in
a subprocess.
"""

import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("opentelemetry.sdk")

_SCRIPT = textwrap.dedent('''
    import json
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry import trace

    from dspy_monty_interpreter import MontyInterpreter, instrument

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    first, second = instrument(), instrument()

    interp = MontyInterpreter(tools={"f": lambda: 1}, output_fields=[{"name": "a"}])
    interp.execute("x = f()")
    try:
        interp.execute("1 / 0")
    except Exception:
        pass
    interp.execute("SUBMIT(x)")
    interp.shutdown()

    spans = exporter.get_finished_spans()
    by_id = {s.context.span_id: s for s in spans}
    print(json.dumps({
        "installed": [first, second],
        "execute": [dict(s.attributes) for s in spans if s.name == "monty.execute"],
        "call_parents": sorted({by_id[s.parent.span_id].name for s in spans
                                if s.name.startswith("call") and s.parent and s.parent.span_id in by_id}),
    }))
''')


def test_execute_spans_and_monty_telemetry():
    import json

    out = subprocess.run([sys.executable, "-c", _SCRIPT], capture_output=True, text=True, check=True)
    data = json.loads(out.stdout.strip().splitlines()[-1])
    assert data["installed"] == [True, False]
    outcomes = [(a["monty.outcome"], a.get("monty.error.type")) for a in data["execute"]]
    assert outcomes == [("none", None), ("error", "CodeExecutionError"), ("submit", None)]
    assert data["call_parents"] == ["run code"]
