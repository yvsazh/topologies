"""Atomic file and directory writes: readers never observe a half-written file."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:  # e.g. Windows: directories cannot be opened
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_save(path: str | Path, writer: Callable[[Path], None], *, fsync: bool = False) -> Path:
    """Call ``writer(tmp_path)`` on a sibling temp file, then ``os.replace`` it into place.

    The temp file is always removed on failure. ``fsync`` also flushes the file
    and its directory (use it for manifests that must survive a crash).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    tmp = Path(name)
    try:
        writer(tmp)
        if fsync:
            with open(tmp, "rb") as fh:
                os.fsync(fh.fileno())
        os.replace(tmp, path)
        if fsync:
            _fsync_dir(path.parent)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def atomic_write_bytes(path: str | Path, data: bytes, *, fsync: bool = False) -> Path:
    return atomic_save(path, lambda tmp: tmp.write_bytes(data), fsync=fsync)


def atomic_write_text(
    path: str | Path, text: str, *, encoding: str = "utf-8", fsync: bool = False
) -> Path:
    return atomic_write_bytes(path, text.encode(encoding), fsync=fsync)


def atomic_write_json(
    path: str | Path, obj: Any, *, fsync: bool = False, **dump_kwargs: Any
) -> Path:
    """``json.dumps(obj, **dump_kwargs)`` written atomically."""
    return atomic_write_text(path, json.dumps(obj, **dump_kwargs), fsync=fsync)


def atomic_dir(
    target: str | Path, writer: Callable[[Path], None], *, keep_existing: bool = True
) -> Path:
    """Build a directory with ``writer(tmp_dir)``, then rename it to ``target``.

    With ``keep_existing`` a concurrent writer that finished first wins (its
    directory is kept). Otherwise an existing ``target`` is replaced (moved
    aside first and restored if the swap fails). Temp directories are removed.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    backup: Path | None = None
    try:
        writer(tmp)
        if target.exists() and not keep_existing:
            backup = Path(tempfile.mkdtemp(prefix=f".{target.name}.old.", dir=target.parent))
            backup.rmdir()
            os.replace(target, backup)
        try:
            os.replace(tmp, target)
        except OSError:
            if backup is not None:
                os.replace(backup, target)
                backup = None
            if not (keep_existing and target.exists()):
                raise
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        if backup is not None and backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
    return target
