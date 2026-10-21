"""Memory-mapped token streams written from a streaming corpus.

``write_token_stream`` encodes documents in batches and writes one directory
another process can reopen with ``TokenStream.open``. The write goes to a
temporary directory and is renamed into place, so two trials that pick the
same destination do not tear the file.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from topologies.data.corpus import TextCorpus


def _tokenizer_fingerprint(tokenizer: Any) -> str:
    payload: dict[str, Any] = {
        "type": type(tokenizer).__name__,
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0) or 0),
        "eos_id": getattr(tokenizer, "eos_id", None),
        "bos_id": getattr(tokenizer, "bos_id", None),
    }
    state = getattr(tokenizer, "_state", None)
    if callable(state):
        payload["state"] = state()
    raw = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def _dtype_for_vocab(vocab_size: int) -> np.dtype:
    """uint16 while every id fits in 16 bits; uint32 from vocab 65536 up."""
    if int(vocab_size) < 65536:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


def _encode_batch_job(spec: tuple[str, list[str]]) -> list[list[int]]:
    """Spawn-safe batch encode. ``spec`` is ``(tokenizer_dir, texts)``."""
    tok_dir, texts = spec
    from topologies.tokenizers.base import Tokenizer

    tokenizer = Tokenizer.load(tok_dir)
    return tokenizer.encode_batch(texts)


class TokenStream:
    """A read-only memmap of token ids plus per-document offsets and labels."""

    def __init__(
        self,
        directory: str | Path,
        tokens: np.memmap,
        doc_offsets: np.ndarray,
        doc_labels: np.ndarray | None,
        label_names: list[str],
        meta: dict[str, Any],
    ) -> None:
        self.directory = Path(directory)
        self.tokens = tokens
        self.doc_offsets = doc_offsets
        self.doc_labels = doc_labels
        self.label_names = list(label_names)
        self.meta = dict(meta)

    @classmethod
    def open(cls, path: str | Path) -> TokenStream:
        directory = Path(path)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        n_tokens = int(meta["n_tokens"])
        tokens = np.memmap(
            directory / "tokens.bin", dtype=np.dtype(meta["dtype"]), mode="r", shape=(n_tokens,)
        )
        offsets = np.fromfile(directory / "docs.idx", dtype=np.int64)
        labels = None
        names: list[str] = []
        label_path = directory / "labels.idx"
        if label_path.is_file():
            labels = np.fromfile(label_path, dtype=np.uint16)
            names = json.loads((directory / "label_names.json").read_text(encoding="utf-8"))
        return cls(directory, tokens, offsets, labels, names, meta)


def write_token_stream(
    corpus: TextCorpus,
    tokenizer: Any,
    path: str | Path,
    *,
    add_eos: bool = True,
    batch_docs: int = 1024,
    workers: int = 1,
) -> TokenStream:
    """Encode ``corpus`` and write a ``TokenStream`` directory.

    An existing complete stream at ``path`` is reopened instead of rewritten.
    ``workers > 1`` saves the tokenizer and encodes batches in subprocesses
    (spawned: call it from code under ``if __name__ == "__main__":``). A
    tokenizer without ``save`` is encoded in this process.
    """
    if batch_docs < 1:
        raise ValueError(f"batch_docs must be >= 1; got {batch_docs}.")
    if workers < 1:
        raise ValueError(f"workers must be >= 1; got {workers}.")
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    lock_path = dest.parent / f".{dest.name}.lock"
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if (dest / "meta.json").is_file():
                return TokenStream.open(dest)
            if dest.exists():
                shutil.rmtree(dest)
            tmp = dest.parent / f".{dest.name}.tmp-{os.getpid()}"
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir()
            try:
                _write_contents(
                    tmp, corpus, tokenizer,
                    add_eos=add_eos, batch_docs=batch_docs, workers=workers,
                )
                os.rename(tmp, dest)
            except Exception:
                shutil.rmtree(tmp, ignore_errors=True)
                raise
            return TokenStream.open(dest)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _write_contents(
    directory: Path,
    corpus: TextCorpus,
    tokenizer: Any,
    *,
    add_eos: bool,
    batch_docs: int,
    workers: int,
) -> None:
    vocab = int(getattr(tokenizer, "vocab_size", 0) or 0)
    dtype = _dtype_for_vocab(vocab)
    limit = int(np.iinfo(dtype).max)
    eos = getattr(tokenizer, "eos_id", None)
    use_eos = bool(add_eos) and eos is not None
    eos_id = int(eos) if use_eos else None
    label_field = getattr(corpus, "label_field", None)

    offsets = [0]
    label_ids: list[int] = []
    label_names: list[str] = []
    label_index: dict[str, int] = {}
    n_bytes = 0
    n_chars = 0
    bytes_per_label: dict[str, int] = {}
    chars_per_label: dict[str, int] = {}
    tokens_per_label: dict[str, int] = {}

    token_path = directory / "tokens.bin"
    with token_path.open("wb") as handle:
        for batch, id_lists in _encoded_batches(
            corpus, tokenizer, directory, batch_docs=batch_docs, workers=workers
        ):
            flat: list[int] = []
            for record, ids in zip(batch, id_lists):
                row = [int(i) for i in ids]
                if eos_id is not None:
                    row.append(eos_id)
                for token_id in row:
                    if token_id < 0 or token_id > limit:
                        raise ValueError(
                            f"Token id {token_id} does not fit in dtype {dtype.name}."
                        )
                flat.extend(row)
                offsets.append(offsets[-1] + len(row))
                text = str(record.get("text") or "")
                extra = 1 if use_eos else 0
                nbytes = len(text.encode("utf-8")) + extra
                nchars = len(text) + extra
                n_bytes += nbytes
                n_chars += nchars
                if label_field:
                    raw_label = record.get(label_field)
                    label = "" if raw_label is None else str(raw_label)
                    if label not in label_index:
                        if len(label_names) >= 65536:
                            raise ValueError("A token stream supports at most 65536 labels.")
                        label_index[label] = len(label_names)
                        label_names.append(label)
                    label_ids.append(label_index[label])
                    bytes_per_label[label] = bytes_per_label.get(label, 0) + nbytes
                    chars_per_label[label] = chars_per_label.get(label, 0) + nchars
                    tokens_per_label[label] = tokens_per_label.get(label, 0) + len(row)
            if flat:
                np.asarray(flat, dtype=dtype).tofile(handle)
                handle.flush()
                _forget_written(handle)

    np.asarray(offsets, dtype=np.int64).tofile(directory / "docs.idx")
    if label_field:
        np.asarray(label_ids, dtype=np.uint16).tofile(directory / "labels.idx")
        (directory / "label_names.json").write_text(
            json.dumps(label_names), encoding="utf-8"
        )
    n_docs = len(offsets) - 1
    fingerprint = ""
    finger = getattr(corpus, "fingerprint", None)
    if callable(finger):
        fingerprint = str(finger())
    meta = {
        "n_tokens": offsets[-1],
        "n_docs": n_docs,
        "n_bytes": n_bytes,
        "n_chars": n_chars,
        "bytes_per_label": bytes_per_label,
        "chars_per_label": chars_per_label,
        "tokens_per_label": tokens_per_label,
        "tokenizer_fingerprint": _tokenizer_fingerprint(tokenizer),
        "corpus_fingerprint": fingerprint,
        "dtype": dtype.name,
        "eos_id": eos_id,
        "vocab_size": vocab,
        "add_eos": bool(use_eos),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (directory / "meta.json").write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")


def _forget_written(handle: Any) -> None:
    """Drop the bytes already flushed to ``handle`` from the page cache."""
    if not hasattr(os, "posix_fadvise"):
        return
    try:
        length = int(handle.tell())
        if length > 0:
            os.posix_fadvise(handle.fileno(), 0, length, os.POSIX_FADV_DONTNEED)
    except (OSError, AttributeError):
        return


def _encoded_batches(
    corpus: TextCorpus,
    tokenizer: Any,
    directory: Path,
    *,
    batch_docs: int,
    workers: int,
) -> Iterator[tuple[list[dict[str, Any]], list[list[int]]]]:
    batches = _record_batches(corpus, batch_docs)
    can_pool = workers > 1 and callable(getattr(tokenizer, "save", None))
    if not can_pool:
        for batch in batches:
            texts = [str(record.get("text") or "") for record in batch]
            yield batch, tokenizer.encode_batch(texts)
        return
    tok_dir = directory / "_tokenizer"
    tokenizer.save(tok_dir)
    try:
        yield from _pool_encode(batches, tok_dir, workers)
    finally:
        shutil.rmtree(tok_dir, ignore_errors=True)


def _pool_encode(
    batches: Iterator[list[dict[str, Any]]],
    tok_dir: Path,
    workers: int,
) -> Iterator[tuple[list[dict[str, Any]], list[list[int]]]]:
    import multiprocessing as mp
    from concurrent.futures.process import BrokenProcessPool

    ctx = mp.get_context("spawn")
    pending: list[tuple[list[dict[str, Any]], Any]] = []
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            for batch in batches:
                texts = [str(record.get("text") or "") for record in batch]
                future = pool.submit(_encode_batch_job, (str(tok_dir), texts))
                pending.append((batch, future))
                if len(pending) >= workers:
                    done, result = pending.pop(0)
                    yield done, result.result()
            for done, result in pending:
                yield done, result.result()
    except BrokenProcessPool as exc:
        raise RuntimeError(
            "write_token_stream(workers>1): a worker process died. Worker processes are "
            "spawned and re-import your main module, so run the call under "
            "`if __name__ == \"__main__\":` (scripts) or use workers=1 (notebooks, stdin)."
        ) from exc


def _record_batches(corpus: TextCorpus, batch_docs: int) -> Iterator[list[dict[str, Any]]]:
    label_field = getattr(corpus, "label_field", None)
    fields: Sequence[str] = ("text",) if not label_field else ("text", label_field)
    batch: list[dict[str, Any]] = []
    for record in corpus.iter_records(fields):
        batch.append(record)
        if len(batch) >= batch_docs:
            yield batch
            batch = []
    if batch:
        yield batch
