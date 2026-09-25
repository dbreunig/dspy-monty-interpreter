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

- `Monty` — subprocess worker pool, created lazily in `_ensure_session()` (entered manually, not via `with`), closed on `shutdown()`. `request_timeout` and `max_processes` are passed through from the adapter constructor.
- `MontySession` — persistent incremental REPL session from `pool.checkout(limits=…, os_policy=…)`. Discarded (returned to pool) on `shutdown()`, on `MontyCrashedError`, on a feed/turn time-limit `TimeoutError`, and whenever RLM resets `_tools_registered = False`.
- `MontySession.feed_run(code, inputs=, external_lookup=, print_callback=, mount=, os=)` — the one execution call. Any signature change here is a breaking change. (`skip_type_check=` and `cwd=` (1.0) also exist but the adapter does not use them. `external_lookup` was named `external_functions` before 0.0.19.)
- `MontyRuntimeError`, `MontySyntaxError` — caught and re-raised as DSPy `CodeExecutionError` / Python `SyntaxError`. `_is_time_limit()` inspects `e.exception()` and `e.display("msg")` to spot Monty's feed/turn limit `TimeoutError` (message contains "time limit exceeded"); a change to that message or type silently disables the session reset.
- `OSPolicy` — passed through as `os_policy` constructor arg (merged over the adapter default `{"sleep": "zero"}`), forwarded to `checkout(os_policy=…)`. New in 1.0; before 1.0 `AbstractOS` clock overrides fired unconditionally, since 1.0 only under `datetime: 'call_host'`.
- `MontyCrashedError` — worker died or hit `request_timeout`; adapter discards the session (state is lost) and re-raises as `CodeInterpreterError`.
- `MountDir` — passed through as `mounts` constructor arg. Keyword-only since 0.0.19 (`host_path=`, `virtual_path=`, `mode=`). Overlay writes are per-feed since 0.0.19 — discarded when each `feed_run` ends.
- `AbstractOS` — passed through as `os_access` constructor arg, forwarded to `feed_run(os=…)`. `OSAccess` is the concrete subclass users most often instantiate.
- `ResourceLimits` — passed through as `resource_limits` constructor arg, forwarded to `checkout(limits=…)`. A TypedDict since 0.0.19. 1.0 replaced `max_duration_secs` with `max_feed_duration_secs` + `max_turn_duration_secs` and added `max_total_sleep_secs`; the README's key list must track this.

Adapter responsibilities Monty does NOT provide:

- Strips markdown ```python fences before execution.
- Injects a `SUBMIT(...)` external function that captures args into a box and returns `None` (so the VM continues past the call). Honored even if a runtime error fires after SUBMIT.
- Wraps every user tool with a callback shim that fires DSPy `on_tool_start` / `on_tool_end` and threads `ACTIVE_CALL_ID`.
- Builds output: print buffer wins over expression value; both stringified.

Project goals (informs the recommendation):

- Stay a **thin** adapter — push capability into Monty, keep wrapping minimal.
- Track real Monty capability: as Monty grows (more stdlib, match stmts) update README's "limitations" list AND `_BASE_EXECUTION_INSTRUCTIONS` in `interpreter.py`. (Classes work as of 0.0.19; still missing as of 1.0: `match`, `yield`, class inheritance, callable `re.sub` replacement.) Verify by installing the new version and running `import <mod>` / syntax probes through `MontyInterpreter().execute()` rather than trusting release notes.
- Maintain compatibility with `dspy>=3.0`'s `CodeInterpreter` protocol.
- Currently pinned: `pydantic-monty>=1.0.0` in `pyproject.toml` (1.0 split the wheel into `pydantic-monty-client` + `pydantic-monty-runtime`; the metapackage pin still works).

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
