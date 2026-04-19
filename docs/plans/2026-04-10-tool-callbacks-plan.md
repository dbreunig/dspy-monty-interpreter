# Tool Callbacks Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fire DSPy's `on_tool_start` / `on_tool_end` callbacks when tools are invoked inside MontyInterpreter, matching ReAct's behavior.

**Architecture:** Wrap each tool callable at `execute()` time with a shim that fires callbacks via `dspy.settings.callbacks` before/after the real function call. Cache `dspy.Tool` instances per tool name to avoid repeated construction.

**Tech Stack:** `dspy.Tool`, `dspy.utils.callback.ACTIVE_CALL_ID`, `dspy.settings`, `uuid`

---

### Task 1: Test callback firing for a single tool call

**Files:**
- Test: `tests/test_interpreter.py`

**Step 1: Write the failing test**

Add to the `# --- Tool dispatch ---` section:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_interpreter.py::test_tool_callback_fires -v`
Expected: FAIL — `len(events) == 0` because callbacks aren't fired yet.

---

### Task 2: Implement `_wrap_tool_with_callbacks`

**Files:**
- Modify: `src/dspy_monty_interpreter/interpreter.py`

**Step 1: Add imports**

At the top of `interpreter.py`, add:

```python
import uuid
import logging

import dspy
from dspy.utils.callback import ACTIVE_CALL_ID
```

**Step 2: Add `_tool_instances` cache to `__init__`**

In `__init__`, after `self._has_state = False`, add:

```python
self._tool_instances: dict[str, dspy.Tool] = {}
```

**Step 3: Add the wrapper method**

Add this method to `MontyInterpreter`, after `_new_repl`:

```python
def _wrap_tool_with_callbacks(self, name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a tool function to fire DSPy on_tool_start/on_tool_end callbacks."""
    def wrapper(**kwargs: Any) -> Any:
        callbacks = dspy.settings.get("callbacks", [])
        if not callbacks:
            return fn(**kwargs)

        # Lazily build and cache a Tool instance for this function
        if name not in self._tool_instances:
            self._tool_instances[name] = dspy.Tool(fn, name=name)
        tool_instance = self._tool_instances[name]

        call_id = uuid.uuid4().hex

        for cb in callbacks:
            try:
                cb.on_tool_start(call_id=call_id, instance=tool_instance, inputs=kwargs)
            except Exception as e:
                logging.getLogger(__name__).warning(f"Callback error on tool start: {e}")

        parent_call_id = ACTIVE_CALL_ID.get()
        ACTIVE_CALL_ID.set(call_id)

        result = None
        exception = None
        try:
            result = fn(**kwargs)
            return result
        except Exception as e:
            exception = e
            raise
        finally:
            ACTIVE_CALL_ID.set(parent_call_id)
            for cb in callbacks:
                try:
                    cb.on_tool_end(call_id=call_id, outputs=result, exception=exception)
                except Exception as e:
                    logging.getLogger(__name__).warning(f"Callback error on tool end: {e}")

    return wrapper
```

**Step 4: Update `execute()` to use the wrapper**

Replace the `external_fns` block in `execute()`:

```python
        external_fns: dict[str, Callable[..., Any]] = {
            name: self._wrap_tool_with_callbacks(name, fn)
            for name, fn in self._tools.items()
        }
        external_fns["SUBMIT"] = submit_fn
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_interpreter.py::test_tool_callback_fires -v`
Expected: PASS

**Step 6: Run full test suite**

Run: `pytest tests/test_interpreter.py -v`
Expected: All tests PASS

**Step 7: Commit**

```bash
git add src/dspy_monty_interpreter/interpreter.py tests/test_interpreter.py
git commit -m "feat: fire DSPy tool callbacks on tool invocations"
```

---

### Task 3: Test callback firing on tool error

**Files:**
- Test: `tests/test_interpreter.py`

**Step 1: Write the failing test**

```python
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
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/test_interpreter.py::test_tool_callback_fires_on_error -v`
Expected: PASS (the implementation from Task 2 already handles this via the try/finally block)

**Step 3: Commit**

```bash
git add tests/test_interpreter.py
git commit -m "test: verify tool callbacks fire on tool errors"
```

---

### Task 4: Test no callbacks when none configured

**Files:**
- Test: `tests/test_interpreter.py`

**Step 1: Write the test**

```python
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
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/test_interpreter.py::test_tool_no_callbacks_fast_path -v`
Expected: PASS

**Step 3: Run full test suite**

Run: `pytest tests/test_interpreter.py -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add tests/test_interpreter.py
git commit -m "test: verify tool fast path with no callbacks"
```

---

### Task 5: Test ACTIVE_CALL_ID context propagation

**Files:**
- Test: `tests/test_interpreter.py`

**Step 1: Write the test**

```python
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
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/test_interpreter.py::test_tool_callback_sets_active_call_id -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_interpreter.py
git commit -m "test: verify ACTIVE_CALL_ID propagation during tool execution"
```

---

### Task 6: Test tool cache invalidation on tool replacement

**Files:**
- Test: `tests/test_interpreter.py`

**Step 1: Write the test**

```python
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
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/test_interpreter.py::test_tool_callback_cache_updates_on_tool_change -v`
Expected: PASS

**Step 3: Consider clearing cache on `_tools_registered` reset**

In the `_tools_registered` setter, add `self._tool_instances.clear()` alongside the repl reset. This ensures stale Tool objects don't persist when RLM injects fresh tools.

Modify `_tools_registered.setter`:

```python
@_tools_registered.setter
def _tools_registered(self, value: bool) -> None:
    if not value and self._has_state:
        self._repl = self._new_repl()
        self._has_state = False
    if not value:
        self._tool_instances.clear()
    self.__tools_registered = value
```

**Step 4: Run full test suite**

Run: `pytest tests/test_interpreter.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/dspy_monty_interpreter/interpreter.py tests/test_interpreter.py
git commit -m "feat: clear tool instance cache on tools reset"
```
