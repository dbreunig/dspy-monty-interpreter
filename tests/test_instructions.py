"""execution_instructions: the prompt text describing Monty, and checks that its claims hold."""

import pytest
from dspy.primitives.code_interpreter import CodeExecutionError

from dspy_monty_interpreter import MontyInterpreter, MountDir
from dspy_monty_interpreter.interpreter import (
    _AVAILABLE_STDLIB,
    _BASE_EXECUTION_INSTRUCTIONS,
    _UNAVAILABLE_STDLIB_EXAMPLES,
)

# --- execution_instructions (dspy >= 3.3.1) ---


def test_execution_instructions_is_class_attribute_string():
    """RLM reads it off the factory (the class), not an instance."""
    assert isinstance(MontyInterpreter.execution_instructions, str)
    assert MontyInterpreter.execution_instructions.strip()


def test_execution_instructions_describe_monty_limitations():
    text = MontyInterpreter.execution_instructions
    assert "match statements" in text and "yield" in text
    assert "State persists" in text
    for mod in ("re", "json", "math", "datetime", "collections", "itertools", "functools", "dataclasses", "base64"):
        assert f"`{mod}`" in text or f" {mod}," in text or f" {mod}." in text or f" {mod} " in text
    assert "pip" in text or "third-party" in text
    assert "network" in text
    # Monty cannot pass builtin methods as values (max(d, key=d.get) fails).
    assert "key=d.get" in text
    assert " annotationlib." in text
    assert "statistics" in text
    # Monty 1.0 additions and long-standing syntax gaps.
    for mod in ("random", "time", "copy", "pathlib"):
        assert f" {mod}," in text or f" {mod}." in text
    assert "yield" in text
    assert "inheritance" in text


# --- MontyInterpreter.factory() and conditional execution_instructions ---


def test_factory_returns_configured_instances():
    factory = MontyInterpreter.factory(request_timeout=1.5)
    interp = factory()
    assert isinstance(interp, MontyInterpreter)
    assert interp._request_timeout == 1.5
    assert factory() is not interp


def test_factory_carries_execution_instructions():
    """A plain lambda has no execution_instructions, so RLM would fall back
    to an empty prompt section; factory() must expose one."""
    factory = MontyInterpreter.factory()
    assert factory.execution_instructions == MontyInterpreter.execution_instructions
    assert "Monty" in factory.execution_instructions


def test_instructions_without_mounts_say_filesystem_unavailable():
    text = MontyInterpreter.factory().execution_instructions
    assert "filesystem is unavailable" in text
    assert "Mounted" not in text


def test_instructions_with_mounts_describe_paths_modes_and_contents(tmp_path):
    reports = tmp_path / "reports"
    (reports / "q3").mkdir(parents=True)
    (reports / "readme.md").write_text("notes")
    (reports / "q3" / "forecast.txt").write_text("forecast")
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    factory = MontyInterpreter.factory(
        mounts=[
            MountDir(host_path=str(reports), virtual_path="/data", mode="read-only"),
            MountDir(host_path=str(scratch), virtual_path="/scratch", mode="overlay"),
        ]
    )
    text = factory.execution_instructions
    assert "filesystem is unavailable" not in text
    assert "/data (read-only)" in text
    assert "q3/" in text and "readme.md" in text
    assert "/scratch (overlay" in text and "discarded" in text
    assert "(empty)" in text
    assert "os.listdir" in text and "Path" in text
    assert "os.walk" in text
    # The factory's instances carry the same text.
    assert factory().execution_instructions == text


def _mount_line(text: str, virtual_path: str) -> str:
    return next(line for line in text.splitlines() if line.startswith(f"- {virtual_path} "))


def test_instructions_small_dir_lists_every_entry(tmp_path):
    for i in range(20):
        (tmp_path / f"f{i:02d}.txt").write_text("x")
    line = _mount_line(
        MontyInterpreter.factory(
            mounts=MountDir(host_path=str(tmp_path), virtual_path="/d", mode="read-only")
        ).execution_instructions,
        "/d",
    )
    assert "f00.txt" in line and "f19.txt" in line


def test_instructions_large_dir_is_summarized_not_listed(tmp_path):
    for i in range(40):
        (tmp_path / f"f{i:02d}.csv").write_text("x")
    for i in range(3):
        (tmp_path / f"notes{i}.md").write_text("x")
    (tmp_path / "sub").mkdir()
    line = _mount_line(
        MontyInterpreter.factory(
            mounts=MountDir(host_path=str(tmp_path), virtual_path="/d", mode="read-only")
        ).execution_instructions,
        "/d",
    )
    assert "43 files and 1 directory" in line
    assert ".csv ×40" in line and ".md ×3" in line
    assert "e.g. f00.csv" in line
    assert "f39.csv" not in line
    assert len(line) < 320


def test_instructions_truncate_long_names(tmp_path):
    (tmp_path / ("a" * 120 + ".txt")).write_text("x")
    line = _mount_line(
        MontyInterpreter.factory(
            mounts=MountDir(host_path=str(tmp_path), virtual_path="/d", mode="read-only")
        ).execution_instructions,
        "/d",
    )
    assert "a" * 120 not in line
    assert "…" in line
    assert len(line) < 120


def test_instructions_respect_character_budget_per_mount(tmp_path):
    # 20 entries of 40 chars each would be ~850 chars as a plain list.
    for i in range(20):
        (tmp_path / (f"{i:02d}_" + "x" * 37)).write_text("x")
    line = _mount_line(
        MontyInterpreter.factory(
            mounts=MountDir(host_path=str(tmp_path), virtual_path="/d", mode="read-only")
        ).execution_instructions,
        "/d",
    )
    assert len(line) <= 320
    assert "more" in line


def test_instructions_listing_can_be_disabled(tmp_path):
    (tmp_path / "secret.txt").write_text("x")
    text = MontyInterpreter.factory(
        mounts=MountDir(host_path=str(tmp_path), virtual_path="/d", mode="read-only"),
        mount_listing_limit=0,
    ).execution_instructions
    assert "- /d (read-only)" in text
    assert "secret.txt" not in text
    assert "containing" not in text


def test_instructions_listing_limit_is_configurable(tmp_path):
    for i in range(5):
        (tmp_path / f"f{i}.txt").write_text("x")
    text = MontyInterpreter(
        mounts=MountDir(host_path=str(tmp_path), virtual_path="/d", mode="read-only"),
        mount_listing_limit=3,
    ).execution_instructions
    assert "5 files" in text and "f4.txt" not in text


def test_instructions_scan_is_bounded(tmp_path):
    import dspy_monty_interpreter.interpreter as mod

    for i in range(mod._MOUNT_SCAN_LIMIT + 5):
        (tmp_path / f"{i}.txt").write_text("x")
    line = _mount_line(
        MontyInterpreter.factory(
            mounts=MountDir(host_path=str(tmp_path), virtual_path="/d", mode="read-only")
        ).execution_instructions,
        "/d",
    )
    assert f"{mod._MOUNT_SCAN_LIMIT:,}+ entries" in line


def test_instructions_tolerate_unreadable_host_path(tmp_path):
    """MountDir requires the host path to exist, but it can vanish before
    the factory is built; describe the mount anyway."""
    later = tmp_path / "later"
    later.mkdir()
    mount = MountDir(host_path=str(later), virtual_path="/out", mode="read-write")
    later.rmdir()
    text = MontyInterpreter.factory(mounts=mount).execution_instructions
    assert "/out (read-write" in text
    assert "not readable" in text


def test_rlm_prompt_includes_mount_description(tmp_path):
    import dspy

    (tmp_path / "a.csv").write_text("x")
    factory = MontyInterpreter.factory(
        mounts=MountDir(host_path=str(tmp_path), virtual_path="/data", mode="read-only")
    )
    rlm = dspy.RLM("q -> a", interpreter_factory=factory)
    instructions = rlm.generate_action.signature.instructions
    assert "/data (read-only)" in instructions and "a.csv" in instructions


def test_instructions_separate_filesystem_section_with_newline(tmp_path):
    no_mounts = MontyInterpreter.factory().execution_instructions
    assert "calling it.\nThe filesystem is unavailable" in no_mounts
    assert no_mounts == no_mounts.strip()

    with_mounts = MontyInterpreter.factory(
        mounts=MountDir(host_path=str(tmp_path), virtual_path="/d", mode="read-only")
    ).execution_instructions
    assert "calling it.\nMounted directories" in with_mounts
    assert with_mounts == with_mounts.strip()


# --- execution_instructions claims, checked against the installed Monty ---
#
# The prompt tells the model what Monty can do. These tests fail when a Monty
# upgrade makes a claim stale, so update the prompt and the table together.



@pytest.fixture(scope="module")
def claims_interp():
    interp = MontyInterpreter()
    yield interp
    interp.shutdown()


@pytest.mark.parametrize("module", _AVAILABLE_STDLIB)
def test_listed_stdlib_module_imports(claims_interp, module):
    claims_interp.execute(f"import {module}")


# (module, phrase in the prompt that rules it out)
_UNAVAILABLE_MODULES = [
    *((module, module) for module in _UNAVAILABLE_STDLIB_EXAMPLES),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("requests", "requests"),
    ("subprocess", "no subprocesses"),
    ("socket", "no network access"),
]


@pytest.mark.parametrize(("module", "phrase"), _UNAVAILABLE_MODULES, ids=[m for m, _ in _UNAVAILABLE_MODULES])
def test_unavailable_module_does_not_import(claims_interp, module, phrase):
    assert phrase in _BASE_EXECUTION_INSTRUCTIONS
    with pytest.raises(CodeExecutionError, match="ModuleNotFoundError"):
        claims_interp.execute(f"import {module}")


# (phrase in the prompt, code, None if the code runs or the error it raises)
_SYNTAX_CLAIMS = [
    ("classes", "class P:\n    def __init__(self, x):\n        self.x = x\nP(1).x", None),
    ("decorators", "def deco(f):\n    return f\n@deco\ndef g():\n    return 1\ng()", None),
    ("@dataclass", "from dataclasses import dataclass\n@dataclass\nclass P:\n    x: int\nP(1).x", None),
    ("complex numbers", "1j * 1j", None),
    ("comprehensions", "[x for x in range(3)] + [v for v in {1: 2}.values()]", None),
    ("generator expressions", "sum(x for x in range(3))", None),
    ("lambdas", "(lambda x: x + 1)(1)", None),
    (
        "with statements",
        "class C:\n    def __enter__(self):\n        return 1\n    def __exit__(self, *a):\n        return False\n"
        "with C() as c:\n    v = c\nv",
        None,
    ),
    ("try/except", "try:\n    1 / 0\nexcept ZeroDivisionError:\n    v = 'caught'\nv", None),
    ("match statements", "match 1:\n    case 1:\n        v = 'one'\nv", None),
    ("datetime.now()", "import datetime\ndatetime.datetime.now() is not None", None),
    ("time.time()", "import time\ntime.time() > 0", None),
    ("generator functions (yield)", "def g():\n    yield 1\nlist(g())", "NotImplementedError"),
    ("class inheritance", "class A:\n    pass\nclass B(A):\n    pass", "NotImplementedError"),
    ("subclassing Exception", "class E(Exception):\n    pass", "NotImplementedError"),
    ("callable replacement in re.sub()", "import re\nre.sub('a', lambda m: 'b', 'a')", "TypeError"),
    ("max(d, key=d.get)", "d = {'a': 1}\nmax(d, key=d.get)", "AttributeError"),
    ("sorted(items, key=str.lower)", "sorted(['b', 'A'], key=str.lower)", "AttributeError"),
    ("max(d, key=lambda k: d[k])", "d = {'a': 1, 'b': 2}\nmax(d, key=lambda k: d[k])", None),
    ("max(d.items(), key=lambda kv: kv[1])", "d = {'a': 1, 'b': 2}\nmax(d.items(), key=lambda kv: kv[1])", None),
    ("no environment variables", "import os\nos.getenv('PATH')", "RuntimeError"),
]


@pytest.mark.parametrize(("phrase", "code", "error"), _SYNTAX_CLAIMS, ids=[c[0] for c in _SYNTAX_CLAIMS])
def test_prompt_syntax_claim_holds(claims_interp, phrase, code, error):
    assert phrase in _BASE_EXECUTION_INSTRUCTIONS
    if error is None:
        claims_interp.execute(code)
    else:
        with pytest.raises(CodeExecutionError, match=error):
            claims_interp.execute(code)
