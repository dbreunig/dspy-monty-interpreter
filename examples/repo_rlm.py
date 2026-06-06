"""Example: a Recursive Language Model specialized for analyzing repositories.

This is an EXAMPLE built on top of the library, not part of the package's
public API. It demonstrates how to surface a Monty filesystem overlay to an
LLM driving ``dspy.RLM``.

The core library already enables filesystem access: ``MontyInterpreter`` accepts
``mounts`` / ``os_access`` and forwards them to Monty. The interesting part this
example shows is how to make the *model* aware of the mounted files, since
``dspy.RLM`` treats the interpreter as opaque to its prompt. We do that with two
seams that use only public DSPy API:

1. A manifest spliced into the action instructions via
   ``Signature.with_instructions()`` (the prompt's "Available:" block).
2. A structured ``repos`` REPL variable injected in ``forward()`` / ``aforward()``
   so the model can enumerate paths programmatically.

Repos may be local checkouts or remote specs (a clone URL or ``owner/repo``),
which are shallow-cloned to a temp directory at construction. Clones persist on
disk unless ``cleanup=True`` is passed.

Run it::

    import dspy
    from repo_rlm import RepoRLM

    dspy.configure(lm=dspy.LM("anthropic/claude-opus-4-8"))
    analyzer = RepoRLM("psf/requests", cleanup=True)
    print(analyzer(task="Map the public API and module layout.").report)
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import weakref
from pathlib import Path
from typing import Any, Iterable, Literal

import dspy

from dspy_monty_interpreter import MontyInterpreter, MountDir

# Directory/file names skipped when counting files and building the tree, so the
# model isn't drowned in vendored deps and build artifacts.
DEFAULT_EXCLUDES: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "venv",
        "env",
        "node_modules",
        ".idea",
        ".vscode",
        "dist",
        "build",
        ".eggs",
        ".cache",
    }
)

# Extension -> language label for the predominant-language heuristic.
_EXT_LANG: dict[str, str] = {
    ".py": "Python",
    ".pyi": "Python",
    ".js": "JavaScript",
    ".mjs": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".jsx": "JavaScript",
    ".rs": "Rust",
    ".go": "Go",
    ".java": "Java",
    ".kt": "Kotlin",
    ".rb": "Ruby",
    ".c": "C",
    ".h": "C",
    ".cpp": "C++",
    ".cc": "C++",
    ".hpp": "C++",
    ".cs": "C#",
    ".php": "PHP",
    ".swift": "Swift",
    ".scala": "Scala",
    ".sh": "Shell",
}

# owner/repo (e.g. "psf/requests"), no scheme, no path traversal.
_OWNER_REPO_RE = re.compile(r"^[\w.-]+/[\w.-]+$")

_DEFAULT_SIGNATURE = "task -> report: str"

_MANIFEST_HEADER = "Repositories mounted under {root} ({mode}, read with pathlib):"


class RepoRLM(dspy.RLM):
    """Example RLM specialized for analyzing one or more repositories with Monty.

    Args:
        repos: What to analyze. A path or remote spec, a list of them, or a
            ``{name: path_or_spec}`` dict for explicit mount names. A value is
            treated as a local checkout if it exists on disk, otherwise as a
            remote (clone URL or ``owner/repo``) to shallow-clone.
        signature: RLM signature. Defaults to ``"task -> report: str"``.
        mode: Mount mode for every repo: ``"read-only"`` (default) or
            ``"overlay"`` (reads fall through to the host; writes are captured
            in memory, leaving the host untouched).
        mount_root: Virtual directory the repos are mounted beneath.
        exclude: Directory/file names skipped in the manifest. Defaults to
            :data:`DEFAULT_EXCLUDES`.
        cleanup: If True, delete any cloned temp directories when this instance
            is closed/garbage-collected. Local checkouts are never deleted.
        resource_limits / tools / sub_lm: forwarded to the interpreter / RLM.
        Remaining keyword arguments are forwarded to :class:`dspy.RLM`.

    Example::

        analyzer = RepoRLM("psf/requests")
        print(analyzer(task="Map the public API and module layout.").report)
    """

    def __init__(
        self,
        repos: str | Path | Iterable[str | Path] | dict[str, str | Path],
        signature: Any = _DEFAULT_SIGNATURE,
        *,
        mode: Literal["read-only", "overlay"] = "read-only",
        mount_root: str = "/repos",
        exclude: Iterable[str] | None = None,
        cleanup: bool = False,
        resource_limits: Any = None,
        max_iterations: int = 30,
        max_llm_calls: int = 60,
        max_output_chars: int = 10_000,
        verbose: bool = False,
        tools: list | None = None,
        sub_lm: dspy.LM | None = None,
    ) -> None:
        self._mount_root = "/" + mount_root.strip("/")
        self._mode = mode
        self._excludes = frozenset(exclude) if exclude is not None else DEFAULT_EXCLUDES
        self._cleanup = cleanup
        self._cloned_dirs: list[str] = []
        self._finalizer: weakref.finalize | None = None

        # Resolve every repo to a local host path (cloning remotes as needed),
        # keyed by a unique mount name.
        self.repo_paths: dict[str, str] = self._resolve_repos(repos)

        # The core library already supports filesystem overlay via the
        # constructor: we build the interpreter with the mounts here.
        mounts = [
            MountDir(self._mount_path(name), path, mode=mode)
            for name, path in self.repo_paths.items()
        ]
        interp = MontyInterpreter(
            tools={t.__name__: t for t in tools} if tools else None,
            resource_limits=resource_limits,
            mounts=mounts,
        )

        # Structured index injected as the `repos` REPL variable, plus the prose
        # manifest spliced into the action instructions.
        self.repo_index: dict[str, dict[str, Any]] = self._build_index()
        manifest = self._format_manifest()

        super().__init__(
            signature,
            max_iterations=max_iterations,
            max_llm_calls=max_llm_calls,
            max_output_chars=max_output_chars,
            verbose=verbose,
            tools=tools,
            sub_lm=sub_lm,
            interpreter=interp,
        )

        # Splice the manifest into the action instructions. We operate on the
        # already-built Predict signature via the public with_instructions()
        # rather than overriding RLM's private _build_signatures().
        action_sig = self.generate_action.signature
        self.generate_action.signature = action_sig.with_instructions(
            action_sig.instructions + "\n\n" + manifest
        )

        if self._cleanup and self._cloned_dirs:
            self._finalizer = weakref.finalize(
                self, _remove_dirs, list(self._cloned_dirs)
            )

    # ----- repo resolution -------------------------------------------------

    def _resolve_repos(
        self, repos: str | Path | Iterable[str | Path] | dict[str, str | Path]
    ) -> dict[str, str]:
        if isinstance(repos, dict):
            items: list[tuple[str | None, str | Path]] = list(repos.items())
        elif isinstance(repos, (str, Path)):
            items = [(None, repos)]
        else:
            items = [(None, r) for r in repos]

        resolved: dict[str, str] = {}
        for explicit_name, spec in items:
            host_path = self._resolve_one(spec)
            name = explicit_name or self._derive_name(spec)
            resolved[self._unique_name(name, resolved)] = host_path
        if not resolved:
            raise ValueError("RepoRLM requires at least one repo")
        return resolved

    def _resolve_one(self, spec: str | Path) -> str:
        path = Path(spec).expanduser()
        if path.exists():
            if not path.is_dir():
                raise ValueError(f"Repo path is not a directory: {path}")
            return str(path.resolve())
        if isinstance(spec, str) and self._looks_like_remote(spec):
            return self._clone(spec)
        raise FileNotFoundError(
            f"Repo {spec!r} is not an existing directory and does not look like "
            f"a clone URL or 'owner/repo' spec"
        )

    @staticmethod
    def _looks_like_remote(spec: str) -> bool:
        return (
            spec.startswith(("http://", "https://", "git@", "ssh://"))
            or spec.endswith(".git")
            or bool(_OWNER_REPO_RE.match(spec))
        )

    def _clone(self, spec: str) -> str:
        url = spec
        if _OWNER_REPO_RE.match(spec) and not spec.startswith(
            ("http://", "https://", "git@", "ssh://")
        ):
            url = f"https://github.com/{spec}"
        dest = tempfile.mkdtemp(prefix="repo-rlm-")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", url, dest],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as e:
            shutil.rmtree(dest, ignore_errors=True)
            raise RuntimeError("git is required to clone remote repos") from e
        except subprocess.CalledProcessError as e:
            shutil.rmtree(dest, ignore_errors=True)
            raise RuntimeError(
                f"Failed to clone {url!r}: {e.stderr.strip() or e}"
            ) from e
        self._cloned_dirs.append(dest)
        return dest

    @staticmethod
    def _derive_name(spec: str | Path) -> str:
        text = str(spec).rstrip("/")
        if text.endswith(".git"):
            text = text[: -len(".git")]
        name = text.split("/")[-1] or text
        # Sanitize to something path-safe for the virtual mount.
        name = re.sub(r"[^\w.-]", "_", name)
        return name or "repo"

    @staticmethod
    def _unique_name(name: str, taken: dict[str, str]) -> str:
        if name not in taken:
            return name
        i = 2
        while f"{name}-{i}" in taken:
            i += 1
        return f"{name}-{i}"

    def _mount_path(self, name: str) -> str:
        return f"{self._mount_root}/{name}"

    # ----- manifest --------------------------------------------------------

    def _build_index(self) -> dict[str, dict[str, Any]]:
        index: dict[str, dict[str, Any]] = {}
        for name, host_path in self.repo_paths.items():
            file_count, ext_counts = self._scan(Path(host_path))
            lang = self._predominant_language(ext_counts)
            index[name] = {
                "root": self._mount_path(name),
                "mode": self._mode,
                "files": file_count,
                "language": lang,
                "tree": self._top_level(Path(host_path)),
            }
        return index

    def _scan(self, root: Path) -> tuple[int, dict[str, int]]:
        file_count = 0
        ext_counts: dict[str, int] = {}
        for dirpath, dirnames, filenames in os.walk(root):
            # Prune excluded directories in place so os.walk skips them.
            dirnames[:] = [d for d in dirnames if d not in self._excludes]
            for fn in filenames:
                if fn in self._excludes:
                    continue
                file_count += 1
                ext = os.path.splitext(fn)[1].lower()
                if ext:
                    ext_counts[ext] = ext_counts.get(ext, 0) + 1
        return file_count, ext_counts

    @staticmethod
    def _predominant_language(ext_counts: dict[str, int]) -> str | None:
        best: str | None = None
        best_n = 0
        for ext, n in ext_counts.items():
            lang = _EXT_LANG.get(ext)
            if lang and n > best_n:
                best, best_n = lang, n
        return best

    def _top_level(self, root: Path, limit: int = 20) -> list[str]:
        try:
            children = sorted(
                root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
            )
        except OSError:
            return []
        entries: list[str] = []
        for child in children:
            if child.name in self._excludes:
                continue
            entries.append(child.name + ("/" if child.is_dir() else ""))
            if len(entries) >= limit:
                entries.append("...")
                break
        return entries

    def _format_manifest(self) -> str:
        lines = [_MANIFEST_HEADER.format(root=self._mount_root, mode=self._mode)]
        for name, info in self.repo_index.items():
            meta = []
            if info["language"]:
                meta.append(info["language"])
            meta.append(f"{info['files']} files")
            lines.append(f"- {info['root']}   ({' · '.join(meta)})")
            if info["tree"]:
                lines.append("    " + "  ".join(info["tree"]))
        lines.append(
            "The `repos` variable holds this inventory as structured data. "
            "Read files with pathlib, e.g. Path('/repos/<name>/...').read_text()."
        )
        return "\n".join(lines)

    # ----- execution -------------------------------------------------------

    def forward(self, **input_args: Any) -> dspy.Prediction:
        input_args.setdefault("repos", self.repo_index)
        return super().forward(**input_args)

    async def aforward(self, **input_args: Any) -> dspy.Prediction:
        input_args.setdefault("repos", self.repo_index)
        return await super().aforward(**input_args)

    # ----- lifecycle -------------------------------------------------------

    def close(self) -> None:
        """Release resources; delete cloned temp dirs if ``cleanup=True``."""
        if self._finalizer is not None:
            self._finalizer()
        if self._interpreter is not None:
            self._interpreter.shutdown()

    def __enter__(self) -> RepoRLM:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _remove_dirs(dirs: list[str]) -> None:
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)
