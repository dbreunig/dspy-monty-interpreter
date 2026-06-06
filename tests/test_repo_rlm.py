"""Tests for RepoRLM (construction, manifest, mounts, injection).

These exercise everything except the live-LLM forward() loop, which needs an
API key and is covered by the e2e suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dspy_monty_interpreter import MontyInterpreter, RepoRLM


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """A small fake Python repo with some noise dirs to exclude."""
    repo = tmp_path / "mylib"
    (repo / "src" / "mylib").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "src" / "mylib" / "__init__.py").write_text("VERSION = '1.0'\n")
    (repo / "src" / "mylib" / "core.py").write_text("def run():\n    return 42\n")
    (repo / "tests" / "test_core.py").write_text("def test_run():\n    assert True\n")
    (repo / "README.md").write_text("# mylib\n")
    (repo / "pyproject.toml").write_text("[project]\nname='mylib'\n")
    # Noise that should be excluded from counts and the tree.
    (repo / ".git").mkdir()
    (repo / ".git" / "config").write_text("noise")
    (repo / "__pycache__").mkdir()
    (repo / "__pycache__" / "core.cpython-311.pyc").write_text("noise")
    return repo


def test_single_local_repo_mounts_read_only(sample_repo: Path):
    analyzer = RepoRLM(str(sample_repo))
    interp = analyzer._interpreter

    mounts = interp.mounts
    assert isinstance(mounts, list) and len(mounts) == 1
    # Default signature exposes a `report` output field.
    assert "report" in analyzer.signature.output_fields
    assert analyzer.repo_paths == {"mylib": str(sample_repo.resolve())}


def test_sandbox_can_read_mounted_repo(sample_repo: Path):
    """The mount is live: code in the sandbox can read repo files."""
    analyzer = RepoRLM(str(sample_repo))
    interp = analyzer._interpreter
    out = interp.execute(
        "from pathlib import Path\nPath('/repos/mylib/README.md').read_text()"
    )
    assert out == "# mylib\n"


def test_manifest_in_instructions(sample_repo: Path):
    analyzer = RepoRLM(str(sample_repo))
    instr = analyzer.generate_action.signature.instructions
    assert "/repos/mylib" in instr
    assert "Python" in instr
    assert "read-only" in instr


def test_index_excludes_noise(sample_repo: Path):
    analyzer = RepoRLM(str(sample_repo))
    info = analyzer.repo_index["mylib"]
    # 5 real files: __init__.py, core.py, test_core.py, README.md, pyproject.toml
    assert info["files"] == 5
    assert info["language"] == "Python"
    assert ".git/" not in info["tree"]
    assert "__pycache__/" not in info["tree"]
    assert "README.md" in info["tree"]
    assert "src/" in info["tree"]


def test_repos_variable_injected_on_forward(sample_repo: Path, monkeypatch):
    """forward() should inject the `repos` index as a REPL variable."""
    analyzer = RepoRLM(str(sample_repo))
    captured = {}

    def fake_super_forward(**input_args):
        captured.update(input_args)
        return "sentinel"

    monkeypatch.setattr(
        type(analyzer).__mro__[1], "forward", lambda self, **kw: fake_super_forward(**kw)
    )
    result = analyzer.forward(task="anything")
    assert result == "sentinel"
    assert captured["task"] == "anything"
    assert captured["repos"] == analyzer.repo_index
    assert captured["repos"]["mylib"]["root"] == "/repos/mylib"


def test_multiple_repos_named_and_unique(tmp_path: Path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    for d in (a, b):
        d.mkdir()
        (d / "main.py").write_text("x = 1\n")
    analyzer = RepoRLM({"frontend": str(a), "backend": str(b)})
    assert set(analyzer.repo_paths) == {"frontend", "backend"}
    assert analyzer.repo_index["frontend"]["root"] == "/repos/frontend"
    assert analyzer.repo_index["backend"]["root"] == "/repos/backend"


def test_duplicate_basenames_deduped(tmp_path: Path):
    a = tmp_path / "x" / "lib"
    b = tmp_path / "y" / "lib"
    for d in (a, b):
        d.mkdir(parents=True)
        (d / "f.py").write_text("1\n")
    analyzer = RepoRLM([str(a), str(b)])
    assert set(analyzer.repo_paths) == {"lib", "lib-2"}


def test_overlay_mode_allows_scratch_writes(sample_repo: Path):
    analyzer = RepoRLM(str(sample_repo), mode="overlay")
    interp = analyzer._interpreter
    interp.execute(
        "from pathlib import Path\nPath('/repos/mylib/notes.md').write_text('hi')"
    )
    assert interp.execute("Path('/repos/mylib/notes.md').read_text()") == "hi"
    # Host untouched.
    assert not (sample_repo / "notes.md").exists()


def test_missing_path_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        RepoRLM(str(tmp_path / "does-not-exist"))


def test_custom_signature(sample_repo: Path):
    analyzer = RepoRLM(
        str(sample_repo),
        "task -> public_api: list[str], summary: str",
    )
    assert set(analyzer.signature.output_fields) == {"public_api", "summary"}


def test_provided_interpreter_is_used(sample_repo: Path):
    interp = MontyInterpreter()
    analyzer = RepoRLM(str(sample_repo), interpreter=interp)
    assert analyzer._interpreter is interp
    assert interp.mounts is not None
