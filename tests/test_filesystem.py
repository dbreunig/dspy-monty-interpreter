"""Host directories mounted into the sandbox."""

import pytest
from dspy.primitives.code_interpreter import CodeInterpreterError

from dspy_monty_interpreter import MontyInterpreter, MountDir


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
