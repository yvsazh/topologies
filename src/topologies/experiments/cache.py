"""Shared, content-addressed cache for experiment artifacts (tokenizers, datasets).

Lives at ``<root>/.cache/`` and is shared across experiments: keys include
data fingerprints, so two experiments over the same corpus reuse one
trained tokenizer. Disk writes go to a temporary directory first and are
moved into place atomically (safe with parallel workers).

Fingerprints (best effort, documented in docs/experiments.md):

- documents (list of str): count + sha256 over all documents (streamed);
- file paths: path + size + mtime;
- datasets: length + class + hash of the first and last 8 samples.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from topologies._util.fs import atomic_dir, atomic_save


def _hash_update(h: Any, obj: Any) -> None:
    if isinstance(obj, torch.Tensor):
        t = obj.detach().cpu().contiguous()
        h.update(f"T{tuple(t.shape)}{t.dtype}".encode())
        try:
            h.update(t.numpy().tobytes() if t.numel() <= 1_000_000 else
                     t.flatten()[:1000].numpy().tobytes())
        except Exception:  # noqa: BLE001 - e.g. bfloat16 has no numpy dtype
            h.update(repr(t.float().flatten()[:1000].tolist()).encode())
    elif isinstance(obj, (list, tuple)):
        h.update(f"L{len(obj)}".encode())
        for x in obj:
            _hash_update(h, x)
    elif isinstance(obj, dict):
        for k in sorted(obj, key=str):
            h.update(str(k).encode())
            _hash_update(h, obj[k])
    elif isinstance(obj, (str, bytes)):
        h.update(obj.encode("utf-8", "surrogatepass") if isinstance(obj, str) else obj)
    else:
        h.update(repr(obj)[:2000].encode())


def fingerprint_data(obj: Any) -> str:
    """Stable short hash identifying a data source (see module docstring)."""
    h = hashlib.sha256()
    if obj is None:
        h.update(b"none")
    elif isinstance(obj, (str, Path)) and _is_path(obj):
        p = Path(obj)
        st = p.stat()
        h.update(f"path:{p.resolve()}:{st.st_size}:{int(st.st_mtime)}".encode())
    elif isinstance(obj, (list, tuple)) and all(isinstance(x, (str, Path)) for x in obj) and \
            obj and all(_is_path(x) for x in obj):
        for x in obj:
            h.update(fingerprint_data(x).encode())
    elif isinstance(obj, (list, tuple)) and all(isinstance(x, str) for x in obj):
        h.update(f"docs:{len(obj)}".encode())
        for d in obj:
            h.update(d.encode("utf-8", "surrogatepass"))
            h.update(b"\x00")
    elif isinstance(obj, str):
        h.update(b"text:")
        h.update(obj.encode("utf-8", "surrogatepass"))
    else:
        try:
            n = len(obj)
        except Exception:  # noqa: BLE001
            n = -1
        h.update(f"ds:{type(obj).__module__}.{type(obj).__qualname__}:{n}".encode())
        if n > 0:
            idx = sorted(set(list(range(min(8, n))) + list(range(max(0, n - 8), n))))
            for i in idx:
                try:
                    _hash_update(h, obj[i])
                except Exception:  # noqa: BLE001
                    h.update(f"?{i}".encode())
    return h.hexdigest()[:16]


def _is_path(x: Any) -> bool:
    if isinstance(x, Path):
        return x.exists()
    if not isinstance(x, str) or "\n" in x or not x or len(x) > 1024:
        return False
    try:
        return Path(x).exists()
    except (OSError, ValueError):
        return False


def stable_hash(obj: Any, n: int = 16) -> str:
    """sha256 of the canonical JSON of ``obj`` (sorted keys)."""
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:n]


class Cache:
    """In-process LRU + on-disk store under ``<root>/.cache``.

    ``get_or_create(key, fn)`` memoizes in memory; ``tokenizer(key, fn)``
    and ``tensor(key, fn)`` also persist to disk.
    """

    def __init__(self, root: str | Path | None, *, max_items: int = 16) -> None:
        self.dir = Path(root) / ".cache" if root is not None else None
        self.max_items = max(1, int(max_items))
        self._mem: OrderedDict[str, Any] = OrderedDict()

    # -- memory --------------------------------------------------------
    def get_or_create(self, key: str, fn: Callable[[], Any]) -> Any:
        if key in self._mem:
            self._mem.move_to_end(key)
            return self._mem[key]
        value = fn()
        self._mem[key] = value
        while len(self._mem) > self.max_items:
            self._mem.popitem(last=False)
        return value

    def clear_memory(self) -> None:
        self._mem.clear()

    # -- disk helpers -----------------------------------------------------
    def _atomic_dir(self, target: Path, write: Callable[[Path], None]) -> None:
        # Another worker that finished first wins: keep theirs.
        atomic_dir(target, write, keep_existing=True)

    # -- tokenizers --------------------------------------------------------
    def tokenizer(self, key: str, train: Callable[[], Any]) -> tuple[Any, dict]:
        """``(tokenizer, meta)``; ``meta["train_s"]`` is the original training time."""
        mem_key = f"tok:{key}"
        if mem_key in self._mem:
            self._mem.move_to_end(mem_key)
            return self._mem[mem_key]
        tok, meta = None, {}
        target = self.dir / "tokenizers" / key if self.dir is not None else None
        if target is not None and (target / "tokenizer.json").is_file():
            try:
                from topologies.tokenizers import load_tokenizer

                tok = load_tokenizer(target)
                meta = json.loads((target / "meta.json").read_text()) \
                    if (target / "meta.json").is_file() else {}
                meta["cached"] = True
            except Exception:  # noqa: BLE001 - corrupt cache entry: retrain
                tok = None
        if tok is None:
            t0 = time.perf_counter()
            tok = train()
            meta = {"train_s": time.perf_counter() - t0, "cached": False}
            if target is not None and callable(getattr(tok, "save", None)):
                def _write(d: Path, tok: Any = tok, meta: dict = meta) -> None:
                    tok.save(d)
                    (d / "meta.json").write_text(json.dumps(
                        {k: v for k, v in meta.items() if k != "cached"}))

                try:
                    self._atomic_dir(target, _write)
                except Exception:  # noqa: BLE001, S110 - cache is an optimization
                    pass
        self._mem[mem_key] = (tok, meta)
        while len(self._mem) > self.max_items:  # same LRU bound as tensor()
            self._mem.popitem(last=False)
        return tok, meta

    # -- tensors (tokenized datasets) ---------------------------------------
    def tensor(self, key: str, build: Callable[[], torch.Tensor], *,
               disk: bool = True) -> torch.Tensor:
        mem_key = f"tensor:{key}"
        if mem_key in self._mem:
            self._mem.move_to_end(mem_key)
            return self._mem[mem_key]
        value = None
        path = self.dir / "datasets" / f"{key}.pt" if (self.dir is not None and disk) else None
        if path is not None and path.is_file():
            try:
                value = torch.load(path, map_location="cpu", weights_only=True)
            except Exception:  # noqa: BLE001
                value = None
        if value is None:
            value = build()
            if path is not None and isinstance(value, torch.Tensor):
                try:
                    atomic_save(path, lambda tmp: torch.save(value, tmp))
                except OSError:  # the disk cache is an optimisation; memory copy still works
                    pass
        self._mem[mem_key] = value
        while len(self._mem) > self.max_items:
            self._mem.popitem(last=False)
        return value

    # -- maintenance -----------------------------------------------------
    def entries(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if self.dir is None or not self.dir.is_dir():
            return out
        for kind in ("tokenizers", "datasets"):
            base = self.dir / kind
            if not base.is_dir():
                continue
            for p in sorted(base.iterdir()):
                if p.name.startswith("."):
                    continue
                size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) \
                    if p.is_dir() else p.stat().st_size
                out.append({"kind": kind[:-1], "key": p.stem if p.is_file() else p.name,
                            "bytes": size, "path": str(p)})
        return out

    def clear(self) -> int:
        n = len(self.entries())
        if self.dir is not None and self.dir.is_dir():
            shutil.rmtree(self.dir, ignore_errors=True)
        self._mem.clear()
        return n


__all__ = ["Cache", "fingerprint_data", "stable_hash"]
