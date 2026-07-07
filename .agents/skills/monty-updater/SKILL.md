---
name: monty-updater
description: Use when checking whether a new pydantic-monty release affects dspy-monty-interpreter — fetches Monty's GitHub releases, compares to the pinned version in pyproject.toml, and recommends what (if anything) to update.
---

# monty-updater

## Purpose

`dspy-monty-interpreter` is a thin adapter that exposes [Monty](https://github.com/pydantic/monty) (a Rust-built sandboxed Python interpreter, distributed on PyPI as `pydantic-monty`) as a DSPy `CodeInterpreter`. When Monty ships a new release, this skill audits whether the adapter needs changes — bug fixes, version bumps, new features to surface, or breaking-change handling.

## What the adapter actually does (so you can judge release impact)

Single class: `MontyInterpreter` in `src/dspy_monty_interpreter/interpreter.py`. Conforms to DSPy's `CodeInterpreter` protocol (`tools`, `output_fields`, `_tools_registered`, `start`, `execute`, `shutdown`).

Key Monty surface area it consumes (from `pydantic_monty`):

- `MontyRepl` — persistent incremental REPL. Created in `_new_repl()`, replaced on `shutdown()` and whenever RLM resets `_tools_registered = False`.
- `MontyRepl.feed_run(code, inputs=, external_functions=, print_callback=, mount=, os=)` — the one execution call. Any signature change here is a breaking change. (`skip_type_check=` also exists but adapter does not use it.)
- `MontyRuntimeError`, `MontySyntaxError` — caught and re-raised as DSPy `CodeInterpreterError` / Python `SyntaxError`.
- `MountDir` — passed through as `mounts` constructor arg. (Renamed from `MountDirectory` in 0.0.13.)
- `AbstractOS` — passed through as `os_access` constructor arg, forwarded to `feed_run(os=…)`. `OSAccess` is the concrete subclass users most often instantiate.
- `ResourceLimits` — passed through as `resource_limits` constructor arg, forwarded to `MontyRepl(limits=…)`.

Adapter responsibilities Monty does NOT provide:

- Strips markdown ```python fences before execution.
- Injects a `SUBMIT(...)` external function that captures args into a box and returns `None` (so the VM continues past the call). Honored even if a runtime error fires after SUBMIT.
- Wraps every user tool with a callback shim that fires DSPy `on_tool_start` / `on_tool_end` and threads `ACTIVE_CALL_ID`.
- Builds output: print buffer wins over expression value; both stringified.

Project goals (informs the recommendation):

- Stay a **thin** adapter — push capability into Monty, keep wrapping minimal.
- Track real Monty capability: as Monty grows (classes, more stdlib, match stmts) update README's "limitations" list.
- Maintain compatibility with `dspy>=3.0`'s `CodeInterpreter` protocol.
- Currently pinned: `pydantic-monty>=0.0.15` in `pyproject.toml`.

## Workflow

1. **Read the current pin.** Open `pyproject.toml`, find the `pydantic-monty>=X.Y.Z` line. Record `X.Y.Z`.
2. **Fetch releases.** `WebFetch` https://github.com/pydantic/monty/releases. If that's noisy, also try the GitHub API: `https://api.github.com/repos/pydantic/monty/releases` (no auth needed for public reads, but rate-limited).
3. **Identify unreviewed releases.** List every release with a tag `> X.Y.Z`. If none, report "up to date" and stop.
4. **Pull each release's notes.** For each new tag, read its body. Categorize each bullet as:
   - **Breaking** — signature, exception, or behavior change in any symbol the adapter imports (see surface-area list above). `feed_run` signature changes are the highest-risk class.
   - **Bug fix** — may let us delete an adapter workaround, or fix a known issue in our test suite.
   - **New capability** — new builtins, syntax support (classes, match), new stdlib modules, new `MontyRepl` features, new `ResourceLimits` knobs, new mount options. These usually warrant README updates and sometimes new constructor params.
   - **Internal** — Rust refactors, perf, no Python-visible change. Note but no action.
5. **Cross-check against the adapter.** For each Breaking/New item, grep the relevant symbol in `src/dspy_monty_interpreter/` and `tests/`. State the exact file:line that would change.
6. **Recommend.** Produce a punch list for the user with these sections (omit empty sections):
   - **Required changes** (breaking-change fixes, version-pin bump)
   - **Suggested enhancements** (surface a new Monty feature through the adapter)
   - **Docs/README updates** (limitation list changes, version requirement bumps)
   - **No action** (changes that don't affect us, with one-line reasons)
   Include the proposed new pin (e.g. `pydantic-monty>=A.B.C`) and whether this warrants a `dspy-monty-interpreter` patch/minor/major bump.

**Do not edit code or bump versions in this skill.** Stop at the recommendation. The user decides; release work runs through the `release` skill.

## Quick reference

| Thing to check | Where |
|---|---|
| Current pin | `pyproject.toml` line ~24 |
| Adapter surface area | `src/dspy_monty_interpreter/interpreter.py` |
| Stated limitations | `README.md` (top section) |
| Tests that exercise Monty behavior | `tests/test_interpreter.py` |
| Monty releases | https://github.com/pydantic/monty/releases |
| Monty PyPI metadata | https://pypi.org/pypi/pydantic-monty/json |

## Common mistakes

- **Skipping the surface-area check.** A release note saying "added match statement support" sounds neutral but should trigger a README limitations-list edit.
- **Treating any `MontyRepl` change as breaking.** Only changes to symbols listed in the surface-area section above affect us. Internal Monty changes are noise.
- **Forgetting intermediate releases.** If we're on 0.0.10 and latest is 0.0.13, review 0.0.11, 0.0.12, AND 0.0.13 — a feature added in .11 and removed in .13 still matters for our changelog narrative.
- **Recommending a version bump without checking `requires-python`.** If Monty raises its Python floor, our `pyproject.toml` `requires-python = ">=3.10"` may need to follow.
