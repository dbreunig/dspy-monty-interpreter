"""Synchronous access to Monty's async worker pools.

``AsyncMontyRemote`` reaches sandboxes on another machine (a
``monty-server``), but Monty offers it only as an async API, while DSPy calls
interpreters synchronously. ``SyncPool`` runs an async pool on a private
event loop thread and exposes the subset of the sync ``Monty`` /
``MontySession`` API the interpreter uses.

Each feed runs as a task created in the calling thread's context, so host
callbacks (tools, ``llm_query``, sub-agents) see that thread's ``dspy``
settings. Host callbacks stay synchronous, because Monty turns an async host
function into a coroutine sandbox code would have to await; they run on the
loop thread, so threads sharing one interpreter take turns at host calls.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable, Iterator, Mapping
from typing import Any, TypeVar

T = TypeVar("T")


class _LoopThread:
    """An event loop running on its own daemon thread."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, name="monty-async-pool", daemon=True)
        self._thread.start()

    def run(self, make: Callable[[], Awaitable[T]]) -> T:
        """Await ``make()`` on the loop and wait for the result. ``make`` is
        called there, since Monty binds its awaitables to the running loop;
        the task starts in a copy of the caller's context."""

        async def run() -> T:
            return await make()

        return asyncio.run_coroutine_threadsafe(run(), self.loop).result()

    def stop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5)
        if not self._thread.is_alive():
            self.loop.close()


class SyncSession:
    """A sync facade over an ``AsyncMontySession``."""

    def __init__(self, loop: _LoopThread, session: Any) -> None:
        self._loop = loop
        self._session = session

    def __enter__(self) -> SyncSession:
        self._loop.run(self._session.__aenter__)
        return self

    def __exit__(self, *args: Any) -> None:
        self._loop.run(lambda: self._session.__aexit__(*args))

    def feed_run(self, code: str, **kwargs: Any) -> Any:
        return self._loop.run(lambda: self._session.feed_run(code, **kwargs))

    def dump(self) -> bytes:
        return self._loop.run(self._session.dump)

    def load_session(self, state: bytes) -> None:
        self._loop.run(lambda: self._session.load_session(state))


class SyncPool:
    """A sync facade over an async Monty pool, such as ``AsyncMontyRemote``."""

    def __init__(self, make_pool: Callable[[], Any]) -> None:
        self._make_pool = make_pool
        self._loop: _LoopThread | None = None
        self._pool: Any = None

    def __enter__(self) -> SyncPool:
        self._loop = _LoopThread()
        # Monty's pools bind to the running loop, so build this one there.
        self._pool = self._loop.run(self._create)
        return self

    async def _create(self) -> Any:
        pool = self._make_pool()
        await pool.__aenter__()
        return pool

    def __exit__(self, *args: Any) -> None:
        loop, pool = self._loop, self._pool
        self._loop = self._pool = None
        if loop is None:
            return
        try:
            if pool is not None:
                loop.run(lambda: pool.__aexit__(*args))
        finally:
            loop.stop()

    def checkout(self, **kwargs: Any) -> SyncSession:
        if self._loop is None:
            raise RuntimeError("The pool is not open.")
        modules = kwargs.get("external_modules")
        if modules is not None:
            kwargs["external_modules"] = _AsyncModules(modules)
        return SyncSession(self._loop, self._pool.checkout(**kwargs))


class _AsyncModules(Mapping):
    """The interpreter's live module mapping, read as an async pool expects:
    a factory there returns an awaitable."""

    def __init__(self, modules: Mapping[str, Any]) -> None:
        self._modules = modules

    def __getitem__(self, name: str) -> Any:
        value = self._modules[name]
        if not callable(value):
            return value

        async def build() -> Any:
            return value()

        return build

    def __iter__(self) -> Iterator[str]:
        return iter(self._modules)

    def __len__(self) -> int:
        return len(self._modules)
