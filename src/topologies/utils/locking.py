"""Inter-process file locks (``file_lock``)."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class RunLocked(Exception):
    pass


class LockTimeout(RunLocked):
    """``file_lock`` could not acquire the lock within ``timeout`` seconds."""


@contextmanager
def file_lock(lock_path: str | Path, *, timeout: float = 30.0, poll: float = 0.05):
    """Inter-process exclusive lock on ``lock_path`` (``fcntl.flock`` on POSIX).

    Blocks up to ``timeout`` seconds (``LockTimeout`` after that). The lock is
    released when the holder exits or dies: the OS drops ``flock`` (POSIX) and
    ``msvcrt.locking`` (Windows) locks of dead processes.
    """
    import time

    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+")  # noqa: SIM115 - closed in finally
    locked = False
    try:
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            try:
                _lock(fh)
                locked = True
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise LockTimeout(f"{path} is locked by another process") from None
                time.sleep(poll)
        try:
            fh.seek(0)
            fh.truncate()
            fh.write(str(os.getpid()))
            fh.flush()
        except OSError:
            pass
        yield path
    finally:
        if locked:
            _unlock(fh)
        fh.close()


def _lock(fh: Any) -> None:
    """Non-blocking exclusive lock; ``OSError`` when another process holds it."""
    try:
        import fcntl
    except ImportError:  # Windows
        import msvcrt

        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        return
    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(fh: Any) -> None:
    try:
        import fcntl
    except ImportError:  # Windows
        import msvcrt

        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        return
    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
