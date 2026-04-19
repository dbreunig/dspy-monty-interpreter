# Tool Callback Firing in MontyInterpreter

## Problem

RLM strips `dspy.Tool` objects down to raw functions via `tool.func` before injecting them into the interpreter. When Monty executes these raw functions, `Tool.__call__` (and its `@with_callbacks` decorator) is bypassed entirely. DSPy's `on_tool_start` / `on_tool_end` callbacks never fire.

ReAct does not have this problem because it calls `Tool.__call__` directly.

## Approach

Wrap each tool callable at `execute()` time so that invocations fire DSPy's callback machinery (`on_tool_start` / `on_tool_end`), matching the behavior of ReAct.

## Design

### Wrapper method

A private method `_wrap_tool_with_callbacks(name, fn)` on `MontyInterpreter` returns a wrapped callable that:

1. Reads `dspy.settings.callbacks` for active callbacks.
2. If no callbacks exist, calls the raw function directly (zero overhead).
3. Otherwise:
   - Generates `call_id` via `uuid.uuid4().hex`.
   - Uses a `dspy.Tool(func=raw_fn, name=tool_name)` as the callback `instance`. Constructed once per tool name and cached.
   - Calls `on_tool_start(call_id, instance, inputs)` on each callback.
   - Sets `ACTIVE_CALL_ID` context var (from `dspy.utils.callback`).
   - Executes the real function.
   - Calls `on_tool_end(call_id, outputs, exception)` on each callback.
   - Restores `ACTIVE_CALL_ID`.

### Change in `execute()`

```python
external_fns: dict[str, Callable[..., Any]] = {
    name: self._wrap_tool_with_callbacks(name, fn)
    for name, fn in self._tools.items()
}
external_fns["SUBMIT"] = submit_fn
```

### Key decisions

- **Real `dspy.Tool` instance** used for callback `instance` parameter so consumers can access `.name`, `.func`, `.args`.
- **Tool instances cached** per tool name to avoid repeated construction.
- **`ACTIVE_CALL_ID` context var** set/restored to match `@with_callbacks` exactly.
- **All injected tools wrapped**, including `llm_query` and `llm_query_batched`.
- **SUBMIT excluded** from wrapping (internal plumbing).
- **No new dependencies** beyond existing `dspy` imports.

### Callback payload

```
on_tool_start(call_id, instance=Tool(name="search", ...), inputs={"query": "..."})
on_tool_end(call_id, outputs="result", exception=None)
```

Matches ReAct's callback output exactly.
