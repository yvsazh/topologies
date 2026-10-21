"""Streaming text corpora. Files are read lazily; nothing is loaded up front.

``TextCorpus`` accepts globs of ``.txt``, ``.jsonl``, ``.jsonl.gz``, and
``.parquet``. A parquet reader needs ``pyarrow`` and is imported only when a
parquet file is actually opened.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import random
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

DEFAULT_SEPARATOR = "<|endoftext|>"


def release_file_cache(path: str | Path) -> None:
    """Ask the OS to drop ``path`` from the page cache. Failures are ignored.

    Streaming a corpus larger than RAM should not pin every byte that was
    read. Training later faults back only the pages it touches.
    """
    if not hasattr(os, "posix_fadvise"):
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass
    finally:
        os.close(fd)


def _as_paths(source: str | Path | Sequence[str | Path]) -> list[str]:
    if isinstance(source, (str, Path)):
        return [str(source)]
    return [str(p) for p in source]


def resolve_files(patterns: Sequence[str]) -> list[Path]:
    """Expand globs and return existing files, sorted by path."""
    found: list[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        matches = sorted(path.parent.glob(path.name)) if not path.exists() else [path]
        if path.exists() and path.is_dir():
            matches = sorted(p for p in path.rglob("*") if p.is_file())
        for match in matches:
            if match.is_file() and match.name != "manifest.json":
                found.append(match.resolve())
    # Stable and unique.
    return sorted(set(found))


class TextCorpus:
    """A lazy collection of documents on disk.

    ``text_field`` is the record field that holds the document (jsonl and
    parquet). ``label_field``, when set, is copied onto each record (for
    example a domain name). ``format="auto"`` picks the reader from the
    suffix. ``.txt`` files are split on ``<|endoftext|>``.
    """

    def __init__(
        self,
        paths: str | Path | Sequence[str | Path],
        *,
        text_field: str = "text",
        label_field: str | None = None,
        format: str = "auto",
        separator: str = DEFAULT_SEPARATOR,
    ) -> None:
        self.patterns = _as_paths(paths)
        self.text_field = text_field
        self.label_field = label_field
        self.format = format
        self.separator = separator
        self.files = resolve_files(self.patterns)
        if not self.files:
            raise FileNotFoundError(f"TextCorpus matched no files: {self.patterns}")

    def fingerprint(self) -> str:
        """Stable id of this file set.

        A nearby ``manifest.json`` replaces per-file size and mtime, so editing
        a document without touching the manifest keeps the id. The selected
        paths are still part of the id: two splits that share one manifest
        must not collide.
        """
        manifest = self._manifest_path()
        if manifest is not None:
            hasher = hashlib.sha256()
            hasher.update(manifest.read_bytes())
            hasher.update(b"\0")
            for path in self.files:
                hasher.update(str(path.resolve()).encode())
                hasher.update(b"\0")
            return hasher.hexdigest()
        hasher = hashlib.sha256()
        for path in self.files:
            stat = path.stat()
            hasher.update(f"{path}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
        return hasher.hexdigest()

    def _manifest_path(self) -> Path | None:
        parents = {path.parent for path in self.files}
        # A manifest next to the files, or one directory up when every file
        # lives under that directory.
        candidates: list[Path] = []
        for parent in parents:
            candidates.append(parent / "manifest.json")
            candidates.append(parent.parent / "manifest.json")
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    def iter_records(self, fields: Sequence[str] | None = None) -> Iterator[dict[str, Any]]:
        """Yield document dicts. ``fields`` limits keys; ``text`` is always available."""
        wanted = None if fields is None else set(fields)
        for path in self.files:
            try:
                for record in _read_file(path, self.text_field, self.label_field, self.format,
                                         self.separator):
                    if wanted is None:
                        yield record
                    else:
                        yield {key: record.get(key) for key in wanted}
            finally:
                release_file_cache(path)

    def iter_texts(self) -> Iterator[str]:
        for record in self.iter_records(("text",)):
            text = record.get("text")
            if isinstance(text, str) and text:
                yield text

    def sample(self, max_bytes: int = 300_000_000, seed: int = 0) -> list[str]:
        """Deterministic byte-capped sample.

        Files are visited in an order shuffled by ``seed``. Documents are
        taken in that order until ``max_bytes`` of UTF-8 is reached. The same
        corpus and seed always return the same texts.
        """
        if max_bytes < 1:
            raise ValueError(f"max_bytes must be >= 1; got {max_bytes}.")
        order = list(self.files)
        random.Random(seed).shuffle(order)
        chosen: list[str] = []
        total = 0
        for path in order:
            try:
                for record in _read_file(path, self.text_field, self.label_field, self.format,
                                         self.separator):
                    text = record.get("text")
                    if not isinstance(text, str) or not text:
                        continue
                    size = len(text.encode("utf-8"))
                    if total + size > max_bytes:
                        return chosen
                    chosen.append(text)
                    total += size
            finally:
                release_file_cache(path)
        return chosen


def _read_file(path: Path, text_field: str, label_field: str | None, fmt: str,
               separator: str) -> Iterator[dict[str, Any]]:
    kind = _kind(path, fmt)
    if kind == "txt":
        yield from _read_text(path, separator, label_field)
    elif kind == "jsonl":
        yield from _read_jsonl(path, text_field, label_field, gzipped=path.name.endswith(".gz"))
    elif kind == "parquet":
        yield from _read_parquet(path, text_field, label_field)
    else:
        raise ValueError(f"Unsupported corpus file {path} (format {kind!r}).")


def _kind(path: Path, fmt: str) -> str:
    if fmt != "auto":
        return fmt
    name = path.name.lower()
    if name.endswith((".jsonl", ".jsonl.gz")):
        return "jsonl"
    if name.endswith(".parquet"):
        return "parquet"
    if name.endswith((".txt", ".txt.gz")):
        return "txt"
    raise ValueError(
        f"Cannot tell the format of {path.name}. Pass format='txt', 'jsonl', or 'parquet'."
    )


def _open_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _read_text(path: Path, separator: str, label_field: str | None) -> Iterator[dict[str, Any]]:
    with _open_text(path) as handle:
        raw = handle.read()
    parts = [raw] if not separator else raw.split(separator)
    for part in parts:
        text = part.strip()
        if text:
            yield _record(text, None, label_field)


def _read_jsonl(path: Path, text_field: str, label_field: str | None,
                *, gzipped: bool) -> Iterator[dict[str, Any]]:
    opener = gzip.open if gzipped else open
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get(text_field)
            if not isinstance(text, str) or not text:
                continue
            label = row.get(label_field) if label_field else None
            yield _record(text, label, label_field)


def _read_parquet(path: Path, text_field: str, label_field: str | None) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Reading parquet corpora needs pyarrow. Install topologies[synth-parquet]."
        ) from exc
    columns = [text_field] + ([label_field] if label_field else [])
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=1024, columns=columns):
        data = batch.to_pydict()
        texts = data.get(text_field, [])
        labels = data.get(label_field, [None] * len(texts)) if label_field else [None] * len(texts)
        for text, label in zip(texts, labels):
            if isinstance(text, str) and text:
                yield _record(text, label, label_field)
    try:
        import pyarrow as pa
    except ImportError:
        return
    pa.default_memory_pool().release_unused()


def _record(text: str, label: Any, label_field: str | None) -> dict[str, Any]:
    row: dict[str, Any] = {"text": text}
    if label_field:
        row[label_field] = None if label is None else str(label)
    return row
