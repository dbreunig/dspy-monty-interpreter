"""Tests for MontyInterpreter."""

import pytest
from dspy.primitives.code_interpreter import CodeInterpreter, CodeInterpreterError, FinalOutput

from dspy_monty_interpreter import MontyInterpreter, MountDir


# --- Protocol conformance ---


def test_implements_protocol():
    interp = MontyInterpreter()
    assert isinstance(interp, CodeInterpreter)


# --- Basic execution ---


def test_expression():
    interp = MontyInterpreter()
    result = interp.execute("1 + 2")
    assert result == "3"


def test_no_output():
    interp = MontyInterpreter()
    result = interp.execute("x = 42", variables={"x": 0})
    assert result is None


def test_print_capture():
    interp = MontyInterpreter()
    result = interp.execute('print("hello")')
    assert result == "hello"


def test_print_multiline():
    interp = MontyInterpreter()
    result = interp.execute('print("a")\nprint("b")')
    assert result == "a\nb"


def test_variable_injection():
    interp = MontyInterpreter()
    result = interp.execute("x + y", variables={"x": 10, "y": 32})
    assert result == "42"


# --- State persistence ---


def test_variable_persists_across_calls():
    interp = MontyInterpreter()
    interp.execute("x = 42")
    result = interp.execute("x + 8")
    assert result == "50"


def test_function_persists_across_calls():
    interp = MontyInterpreter()
    interp.execute("def double(n):\n    return n * 2")
    result = interp.execute("double(21)")
    assert result == "42"


def test_closure_persists_across_calls():
    interp = MontyInterpreter()
    interp.execute('prefix = "Answer: "')
    interp.execute("def fmt(text):\n    return prefix + text")
    result = interp.execute('fmt("42")')
    assert result == "Answer: 42"


def test_multiple_accumulations():
    interp = MontyInterpreter()
    interp.execute("a = 1")
    interp.execute("b = a + 1")
    interp.execute("c = a + b")
    result = interp.execute("a + b + c")
    assert result == "6"


# --- SUBMIT handling ---


def test_submit_kwargs():
    interp = MontyInterpreter()
    result = interp.execute('SUBMIT(answer="42")')
    assert isinstance(result, FinalOutput)
    assert result.output == {"answer": "42"}


def test_submit_single_positional():
    interp = MontyInterpreter()
    result = interp.execute("SUBMIT(42)")
    assert isinstance(result, FinalOutput)
    assert result.output == 42


def test_submit_positional_with_output_fields():
    interp = MontyInterpreter(output_fields=[{"name": "answer"}, {"name": "confidence"}])
    result = interp.execute('SUBMIT("yes", 0.9)')
    assert isinstance(result, FinalOutput)
    assert result.output == {"answer": "yes", "confidence": 0.9}


def test_submit_single_positional_with_one_output_field():
    # SUBMIT(value) with a single declared output field should map the
    # value to that field, not return it bare. Otherwise RLM rejects
    # with "FINAL returned <type>, expected dict with fields: [...]".
    interp = MontyInterpreter(output_fields=[{"name": "quarterly_findings"}])
    result = interp.execute("SUBMIT([1, 2, 3])")
    assert isinstance(result, FinalOutput)
    assert result.output == {"quarterly_findings": [1, 2, 3]}


def test_submit_dict_matching_schema_passes_through():
    # If the model already builds the schema dict and submits it,
    # don't double-wrap.
    interp = MontyInterpreter(output_fields=[{"name": "answer"}])
    result = interp.execute('SUBMIT({"answer": "42"})')
    assert isinstance(result, FinalOutput)
    assert result.output == {"answer": "42"}


def test_submit_no_args():
    interp = MontyInterpreter()
    result = interp.execute("SUBMIT()")
    assert isinstance(result, FinalOutput)
    assert result.output is None


def test_submit_after_state_accumulation():
    interp = MontyInterpreter()
    interp.execute("x = 10")
    interp.execute("y = x * 2")
    result = interp.execute("SUBMIT(answer=x + y)")
    assert isinstance(result, FinalOutput)
    assert result.output == {"answer": 30}


# --- Tool dispatch ---


def test_tool_call():
    call_log = []

    def my_tool(query: str) -> str:
        call_log.append(query)
        return "tool result"

    interp = MontyInterpreter(tools={"my_tool": my_tool})
    result = interp.execute('my_tool(query="hello")')
    assert result == "tool result"
    assert call_log == ["hello"]


def test_tool_call_positional_args():
    # LLM-emitted code commonly calls tools positionally — the wrapper
    # must accept *args, not just **kwargs.
    call_log = []

    def merge(a: str, b: str) -> str:
        call_log.append((a, b))
        return f"{a}+{b}"

    interp = MontyInterpreter(tools={"merge": merge})
    assert interp.execute('merge("x", "y")') == "x+y"
    assert interp.execute('merge("x", b="y")') == "x+y"
    assert call_log == [("x", "y"), ("x", "y")]


def test_tool_result_used_in_code():
    def lookup(key: str) -> str:
        return "found_value"

    interp = MontyInterpreter(tools={"lookup": lookup})
    result = interp.execute('result = lookup(key="x")\nprint(result)')
    assert result == "found_value"


def test_tool_error_propagation():
    def failing_tool() -> str:
        raise ValueError("tool broke")

    interp = MontyInterpreter(tools={"failing_tool": failing_tool})
    with pytest.raises(CodeInterpreterError, match="ValueError"):
        interp.execute("failing_tool()")


def test_tool_error_caught_by_code():
    def failing_tool() -> str:
        raise ValueError("oops")

    interp = MontyInterpreter(tools={"failing_tool": failing_tool})
    result = interp.execute(
        "try:\n    failing_tool()\nexcept ValueError:\n    print('caught')"
    )
    assert result == "caught"


# --- Tool isolation across calls ---


def test_tool_called_once_across_accumulations():
    call_count = [0]

    def counted_tool(x: str) -> str:
        call_count[0] += 1
        return f"result_{call_count[0]}"

    interp = MontyInterpreter(tools={"counted_tool": counted_tool})
    interp.execute('a = counted_tool(x="first")')
    assert call_count[0] == 1

    # MontyRepl persists the bound value of `a` natively — the earlier
    # tool call is not re-invoked because old code never re-runs.
    result = interp.execute("print(a)")
    assert call_count[0] == 1  # NOT called again
    assert result == "result_1"


def test_tool_caching_with_new_calls():
    call_log = []

    def my_tool(x: str) -> str:
        call_log.append(x)
        return f"got_{x}"

    interp = MontyInterpreter(tools={"my_tool": my_tool})
    interp.execute('a = my_tool(x="first")')
    assert call_log == ["first"]

    interp.execute('b = my_tool(x="second")')
    # "first" is not re-called — MontyRepl persists `a` natively.
    assert call_log == ["first", "second"]

    result = interp.execute("print(a + ' ' + b)")
    assert result == "got_first got_second"
    assert call_log == ["first", "second"]


# --- Print suppression during replay ---


def test_old_prints_not_repeated():
    interp = MontyInterpreter()
    result1 = interp.execute('print("first")')
    assert result1 == "first"

    result2 = interp.execute('print("second")')
    assert result2 == "second"  # NOT "first\nsecond"


# --- Error mapping ---


def test_syntax_error():
    interp = MontyInterpreter()
    with pytest.raises(SyntaxError):
        interp.execute("def")


def test_runtime_error():
    interp = MontyInterpreter()
    with pytest.raises(CodeInterpreterError):
        interp.execute("1 / 0")


def test_name_error():
    interp = MontyInterpreter()
    with pytest.raises(CodeInterpreterError):
        interp.execute("undefined_var")


# --- Error recovery ---


def test_failed_code_not_accumulated():
    """MontyRepl preserves partial mutations from failed snippets (Python REPL
    semantics). In this test the error occurs before any mutation, so x is
    unchanged. Note: if the snippet were 'x = 99\\n1/0', x would be 99 after
    the failure — unlike the old replay architecture which would revert to 10."""
    interp = MontyInterpreter()
    interp.execute("x = 10")

    with pytest.raises(CodeInterpreterError):
        interp.execute("undefined_var")

    result = interp.execute("x + 5")
    assert result == "15"


# --- RLM compatibility ---


def test_tools_update_mutates():
    interp = MontyInterpreter()
    interp.tools.update({"new_tool": lambda: "hi"})
    assert "new_tool" in interp.tools


def test_output_fields_settable():
    interp = MontyInterpreter()
    interp.output_fields = [{"name": "answer"}]
    assert interp.output_fields == [{"name": "answer"}]


def test_tools_registered_settable():
    interp = MontyInterpreter()
    interp._tools_registered = True
    assert interp._tools_registered is True
    interp._tools_registered = False
    assert interp._tools_registered is False


def test_tools_registered_reset_clears_state():
    """Setting _tools_registered = False (as RLM does between forward()
    calls) should clear accumulated interpreter state."""
    interp = MontyInterpreter()
    interp.execute("x = 42")

    # Simulate what RLM does at the start of each forward() call
    interp._tools_registered = False  # triggers reset (code_history is non-empty)

    with pytest.raises(CodeInterpreterError):
        interp.execute("x")  # x should no longer exist


def test_tools_registered_no_reset_when_clean():
    """Setting _tools_registered = False on a fresh interpreter should NOT
    clear state — there's nothing to clear."""
    interp = MontyInterpreter()

    # No code has been executed, so this is a no-op
    interp._tools_registered = False

    # Interpreter should still work normally
    result = interp.execute("1 + 1")
    assert result == "2"


# --- Lifecycle ---


def test_context_manager():
    with MontyInterpreter() as interp:
        result = interp.execute("1 + 1")
        assert result == "2"


def test_start_shutdown_idempotent():
    interp = MontyInterpreter()
    interp.start()
    interp.start()
    interp.execute("1 + 1")
    interp.shutdown()
    interp.shutdown()


def test_shutdown_clears_state():
    interp = MontyInterpreter()
    interp.execute("x = 42")
    interp.shutdown()
    with pytest.raises(CodeInterpreterError):
        interp.execute("x")


# --- Cross-forward() / SUBMIT persistence ---


def test_submit_replay_does_not_retrigger():
    """After SUBMIT, a new execute() picks up native REPL state without
    re-triggering the old SUBMIT. State simply persists — there is no replay."""
    call_count = [0]

    def my_tool(prompt: str) -> str:
        call_count[0] += 1
        return f"response_{call_count[0]}"

    interp = MontyInterpreter(tools={"my_tool": my_tool})

    # Simulate forward() #1: two iterations, second calls SUBMIT
    interp.execute('data = my_tool(prompt="q1")')
    assert call_count[0] == 1

    result = interp.execute("SUBMIT(answer=data)")
    assert isinstance(result, FinalOutput)
    assert result.output == {"answer": "response_1"}

    # Simulate forward() #2: state still persists, and new code runs against
    # that state with no re-execution of prior snippets.
    result2 = interp.execute('new_data = my_tool(prompt="q2")\nprint(data + " " + new_data)')
    assert call_count[0] == 2
    assert result2 == "response_1 response_2"


def test_tool_not_recalled_after_submit():
    """Old tools are not re-invoked because MontyRepl persists state natively
    (not because they are cached)."""
    call_log = []

    def llm_query(prompt: str) -> str:
        call_log.append(prompt)
        return f"answer_for_{prompt}"

    interp = MontyInterpreter(tools={"llm_query": llm_query})

    # forward() #1
    interp.execute('x = llm_query(prompt="first")')
    interp.execute("SUBMIT(answer=x)")
    assert call_log == ["first"]

    # forward() #2 — old llm_query call never happens; only "second" is new.
    result = interp.execute('y = llm_query(prompt="second")\nprint(x + " " + y)')
    assert call_log == ["first", "second"]
    assert result == "answer_for_first answer_for_second"


def test_state_persists_across_submit_boundaries():
    """Variables from before SUBMIT should be available after SUBMIT."""
    interp = MontyInterpreter(tools={"my_tool": lambda: "val"})

    interp.execute("a = 1")
    interp.execute("b = a + 1")
    interp.execute("SUBMIT(answer=b)")

    # After SUBMIT, a and b should still be accessible
    result = interp.execute("a + b")
    assert result == "3"


def test_tool_changes_between_calls():
    """Persisted state still works when tools dict is replaced between calls."""
    def tool_v1(x: str) -> str:
        return "v1"

    def tool_v2(x: str) -> str:
        return "v2"

    interp = MontyInterpreter(tools={"my_tool": tool_v1})
    interp.execute('a = my_tool(x="test")')

    # Replace tools entirely (simulates RLM creating fresh tools)
    interp._tools.clear()
    interp._tools["my_tool"] = tool_v2
    interp._tools["new_tool"] = lambda: "new"

    # Old code never re-runs under MontyRepl, so `a` retains v1's return value
    # regardless of the new tool mapping.
    result = interp.execute("print(a)")
    assert result == "v1"


# --- Code fence stripping ---


def test_strip_python_code_fence():
    interp = MontyInterpreter()
    result = interp.execute("```python\n1 + 2\n```")
    assert result == "3"


def test_strip_py_code_fence():
    interp = MontyInterpreter()
    result = interp.execute("```py\n1 + 2\n```")
    assert result == "3"


def test_strip_bare_code_fence():
    interp = MontyInterpreter()
    result = interp.execute("```\n1 + 2\n```")
    assert result == "3"


def test_no_strip_inline_backticks():
    """Backticks that aren't wrapping the entire code should be left alone."""
    interp = MontyInterpreter()
    result = interp.execute("x = 'hello'\nprint(x)")
    assert result == "hello"


# --- Filesystem mounts ---


def test_mount_read_only():
    """Sandboxed code can read files from a read-only mount."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "data.txt").write_text("hello from mount")
        interp = MontyInterpreter(
            mounts=MountDir(virtual_path="/data", host_path=tmpdir, mode="read-only")
        )
        result = interp.execute(
            "from pathlib import Path\nPath('/data/data.txt').read_text()"
        )
        assert result == "hello from mount"


def test_mount_overlay_write():
    """Overlay mount captures writes in memory without modifying the host."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "original.txt").write_text("original")
        interp = MontyInterpreter(
            mounts=MountDir(virtual_path="/data", host_path=tmpdir, mode="overlay")
        )
        result = interp.execute(
            "from pathlib import Path\n"
            "Path('/data/new.txt').write_text('created')\n"
            "Path('/data/new.txt').read_text()"
        )
        assert result == "created"
        # Host filesystem not modified
        assert not Path(tmpdir, "new.txt").exists()


def test_mount_read_only_blocks_write():
    """Read-only mount rejects write operations."""
    import tempfile
    interp = MontyInterpreter(
        mounts=MountDir(virtual_path="/data", host_path=tempfile.mkdtemp(), mode="read-only")
    )
    with pytest.raises(CodeInterpreterError):
        interp.execute(
            "from pathlib import Path\nPath('/data/file.txt').write_text('x')"
        )


def test_mount_overlay_discarded_across_executes():
    """Overlay writes are per-execute (per feed) as of Monty 0.0.19 —
    they are discarded when the execute() call ends."""
    import tempfile
    interp = MontyInterpreter(
        mounts=MountDir(virtual_path="/data", host_path=tempfile.mkdtemp(), mode="overlay")
    )
    interp.execute(
        "from pathlib import Path\nPath('/data/state.txt').write_text('transient')"
    )
    with pytest.raises(CodeInterpreterError, match="FileNotFoundError"):
        interp.execute("Path('/data/state.txt').read_text()")


def test_mount_read_write_persists_across_executes():
    """read-write mounts write through to the host, so state survives
    across execute() calls (the replacement for overlay persistence)."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        interp = MontyInterpreter(
            mounts=MountDir(virtual_path="/data", host_path=tmpdir, mode="read-write")
        )
        interp.execute(
            "from pathlib import Path\nPath('/data/state.txt').write_text('persisted')"
        )
        result = interp.execute("Path('/data/state.txt').read_text()")
        assert result == "persisted"
        assert Path(tmpdir, "state.txt").read_text() == "persisted"


# --- SUBMIT / error edge cases ---


def test_submit_honored_despite_post_submit_error():
    """If code calls SUBMIT then later errors, SUBMIT result is still returned."""
    interp = MontyInterpreter()
    result = interp.execute('SUBMIT(answer="got it")\n1/0')
    assert isinstance(result, FinalOutput)
    assert result.output == {"answer": "got it"}


def test_partial_mutation_persists_on_error():
    """Monty's REPL session preserves partial mutations from failed
    snippets, matching Python REPL semantics."""
    interp = MontyInterpreter()
    interp.execute("x = 1")
    with pytest.raises(CodeInterpreterError):
        interp.execute("x = 99\n1/0")  # x is set before the error
    result = interp.execute("x")
    assert result == "99"  # not reverted to 1


# --- Tool callbacks ---


def test_tool_callback_fires():
    """Tool invocation fires on_tool_start and on_tool_end callbacks."""
    import dspy
    from dspy.utils.callback import BaseCallback

    events = []

    class Recorder(BaseCallback):
        def on_tool_start(self, call_id, instance, inputs):
            events.append(("start", call_id, instance.name, inputs))

        def on_tool_end(self, call_id, outputs, exception=None):
            events.append(("end", call_id, outputs, exception))

    def search(query: str) -> str:
        return f"result for {query}"

    interp = MontyInterpreter(tools={"search": search})

    with dspy.context(callbacks=[Recorder()]):
        result = interp.execute('search(query="python")')

    assert result == "result for python"
    assert len(events) == 2

    kind, start_id, name, inputs = events[0]
    assert kind == "start"
    assert name == "search"
    assert inputs == {"query": "python"}

    kind, end_id, outputs, exc = events[1]
    assert kind == "end"
    assert end_id == start_id
    assert outputs == "result for python"
    assert exc is None


def test_tool_callback_fires_on_error():
    """on_tool_end fires with exception when tool raises."""
    import dspy
    from dspy.utils.callback import BaseCallback

    events = []

    class Recorder(BaseCallback):
        def on_tool_start(self, call_id, instance, inputs):
            events.append(("start", call_id))

        def on_tool_end(self, call_id, outputs, exception=None):
            events.append(("end", call_id, outputs, exception))

    def bad_tool() -> str:
        raise ValueError("boom")

    interp = MontyInterpreter(tools={"bad_tool": bad_tool})

    with dspy.context(callbacks=[Recorder()]):
        with pytest.raises(CodeInterpreterError):
            interp.execute("bad_tool()")

    assert len(events) == 2
    assert events[0][0] == "start"
    assert events[1][0] == "end"
    assert events[1][2] is None  # outputs
    assert isinstance(events[1][3], ValueError)  # exception


def test_tool_callback_sets_active_call_id():
    """ACTIVE_CALL_ID is set during tool execution."""
    import dspy
    from dspy.utils.callback import ACTIVE_CALL_ID, BaseCallback

    captured_ids = []

    class Recorder(BaseCallback):
        def on_tool_start(self, call_id, instance, inputs):
            pass

        def on_tool_end(self, call_id, outputs, exception=None):
            pass

    def spy_tool() -> str:
        captured_ids.append(ACTIVE_CALL_ID.get())
        return "ok"

    interp = MontyInterpreter(tools={"spy_tool": spy_tool})

    with dspy.context(callbacks=[Recorder()]):
        interp.execute("spy_tool()")

    assert len(captured_ids) == 1
    assert captured_ids[0] is not None
    # After execution, ACTIVE_CALL_ID should be restored
    assert ACTIVE_CALL_ID.get() is None


def test_tool_no_callbacks_fast_path():
    """Tools work normally when no callbacks are configured."""
    call_log = []

    def my_tool(x: str) -> str:
        call_log.append(x)
        return "ok"

    interp = MontyInterpreter(tools={"my_tool": my_tool})
    # No dspy.context(callbacks=...) — fast path
    result = interp.execute('my_tool(x="test")')
    assert result == "ok"
    assert call_log == ["test"]


def test_tool_callback_cache_updates_on_tool_change():
    """Cached Tool instances update when the underlying function changes."""
    import dspy
    from dspy.utils.callback import BaseCallback

    instances = []

    class Recorder(BaseCallback):
        def on_tool_start(self, call_id, instance, inputs):
            instances.append(instance)

        def on_tool_end(self, call_id, outputs, exception=None):
            pass

    def tool_v1(x: str) -> str:
        return "v1"

    def tool_v2(x: str) -> str:
        return "v2"

    interp = MontyInterpreter(tools={"my_tool": tool_v1})

    with dspy.context(callbacks=[Recorder()]):
        interp.execute('my_tool(x="a")')

        # Replace tool (as RLM does between forward() calls)
        interp._tools["my_tool"] = tool_v2
        interp._tool_instances.clear()
        interp.execute('my_tool(x="b")')

    assert len(instances) == 2
    assert instances[0].func is tool_v1
    assert instances[1].func is tool_v2


# --- pydantic-monty 0.0.18 regressions ---


def test_comprehension_store_in_repl():
    """REPL store of names bound inside comprehension expressions (monty #297).
    Previously, building a comprehension that referenced persisted state and
    assigning the result could fail under MontyRepl."""
    interp = MontyInterpreter()
    interp.execute("nums = [1, 2, 3, 4]")
    assert interp.execute("squares = [n * n for n in nums]\nprint(sum(squares))") == "30"
    # The comprehension result persists and is reusable on the next call.
    assert interp.execute("len(squares)") == "4"
    # Dict comprehension referencing prior state behaves the same.
    assert interp.execute("m = {n: n * n for n in nums}\nprint(m[3])") == "9"


def test_context_manager_with_open():
    """`with` / context-manager support plus the sandboxed open() builtin
    (monty #462, #456, #461), exercised over an overlay mount. Overlay
    writes are per-execute, so the write/read round-trip happens in one
    execute() call."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "in.txt").write_text("line one\nline two\n")
        interp = MontyInterpreter(
            mounts=MountDir(virtual_path="/data", host_path=tmpdir, mode="overlay")
        )
        assert interp.execute(
            "with open('/data/in.txt') as f:\n    data = f.read()\nprint(len(data))"
        ) == "18"
        assert interp.execute(
            "with open('/data/out.txt', 'w') as f:\n    f.write('hello cm')\n"
            "with open('/data/out.txt') as f:\n    print(f.read())"
        ) == "hello cm"


def test_parallel_execute_thread_isolation():
    """Concurrent execute() calls from different threads get isolated
    sessions: each thread's variables are its own."""
    import threading

    interp = MontyInterpreter()
    barrier = threading.Barrier(2)
    results: dict[str, str] = {}
    errors: list[Exception] = []

    def worker(name: str, value: int) -> None:
        try:
            barrier.wait()
            interp.execute(f"x = {value}")
            barrier.wait()
            results[name] = interp.execute("print(x)")
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=worker, args=("a", 1)),
        threading.Thread(target=worker, args=("b", 2)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert results == {"a": "1", "b": "2"}
    interp.shutdown()


def test_parallel_reset_only_clears_calling_thread():
    """RLM's per-forward reset (_tools_registered = False) discards only
    the calling thread's session, not other threads' state."""
    import threading

    interp = MontyInterpreter()
    interp.execute("x = 'main'")

    def other_forward() -> None:
        # Simulates RLM starting a forward() on another thread.
        interp._tools_registered = False
        interp.execute("x = 'other'")

    t = threading.Thread(target=other_forward)
    t.start()
    t.join()

    # Main thread's state survived the other thread's reset.
    assert interp.execute("print(x)") == "main"
    interp.shutdown()


def test_parallel_evaluate_smoke():
    """Many sequential tasks spread over a thread pool, each doing an
    RLM-style reset + stateful execute sequence, all come back correct."""
    from concurrent.futures import ThreadPoolExecutor

    interp = MontyInterpreter(max_processes=4)

    def task(i: int) -> str:
        interp._tools_registered = False  # RLM does this per forward()
        interp.execute(f"v = {i}")
        interp.execute("v = v * 10")
        return interp.execute("print(v)")

    with ThreadPoolExecutor(max_workers=4) as ex:
        outputs = list(ex.map(task, range(12)))

    assert outputs == [str(i * 10) for i in range(12)]
    interp.shutdown()


def test_shutdown_reclaims_all_thread_sessions():
    """shutdown() discards sessions created by other threads, and the
    interpreter remains usable afterward with fresh state."""
    import threading

    interp = MontyInterpreter()
    ready = threading.Event()
    release = threading.Event()

    def worker() -> None:
        interp.execute("y = 'thread'")
        ready.set()
        release.wait()

    t = threading.Thread(target=worker)
    t.start()
    ready.wait()
    interp.execute("y = 'main'")
    interp.shutdown()
    release.set()
    t.join()

    # Fresh session after shutdown: y is gone.
    with pytest.raises(CodeInterpreterError):
        interp.execute("y")
    interp.shutdown()


def test_user_defined_class():
    """User-defined classes work as of Monty 0.0.19, including state
    that persists across execute() calls."""
    interp = MontyInterpreter()
    interp.execute(
        "class Counter:\n"
        "    def __init__(self):\n"
        "        self.n = 0\n"
        "    def bump(self):\n"
        "        self.n += 1\n"
        "        return self.n\n"
        "c = Counter()"
    )
    assert interp.execute("c.bump()\nc.bump()\nprint(c.bump())") == "3"


def test_worker_crash_raises_and_recovers():
    """A crashed worker (here: exceeding request_timeout) surfaces as
    CodeInterpreterError, and the interpreter recovers with a fresh
    session on the next execute()."""
    interp = MontyInterpreter(request_timeout=0.5)
    interp.execute("x = 1")
    with pytest.raises(CodeInterpreterError):
        interp.execute("i = 0\nwhile True:\n    i += 1")
    # Session was lost with the worker; the next call gets a fresh one.
    result = interp.execute("print('alive')")
    assert result == "alive"


def test_external_function_identity():
    """Name-based identity and equality for external function values
    (monty #458) — covers both injected tools and SUBMIT."""
    def my_tool(x: str) -> str:
        return "v"

    interp = MontyInterpreter(tools={"my_tool": my_tool})
    assert interp.execute("f = my_tool\nprint(f is my_tool)") == "True"
    assert interp.execute("f = my_tool\nprint(f == my_tool)") == "True"
    assert interp.execute("print(SUBMIT is SUBMIT)") == "True"
