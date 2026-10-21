"""Tokenizer statistics on a corpus (compression, unknown rate, round-trip, speed).

    from topologies.tokenizers import tokenizer_stats
    tokenizer_stats(tok, val_docs)
    # {"vocab_size": 4096, "bytes_per_token": 3.9, "chars_per_token": 3.8, ...}

Byte/char/token counts follow ``lm_dataset``: each document is encoded on its
own and, when the tokenizer has an EOS token, one EOS token plus one boundary
byte/char is counted per document. ``bytes_per_token`` is what
``topologies.metrics.bits_per_byte`` needs.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any


def _encode(tok: Any, text: str) -> list[int]:
    if hasattr(tok, "encode"):
        return list(tok.encode(text))
    return list(tok(text))


def corpus_counts(tok: Any, docs: Sequence[str]) -> dict[str, int]:
    """``{"docs", "bytes", "chars", "tokens", "unk"}`` for ``docs`` under ``tok``."""
    has_eos = getattr(tok, "eos_id", None) is not None
    unk = getattr(tok, "unk_id", None)
    n_bytes = n_chars = n_tokens = n_unk = 0
    for d in docs:
        ids = _encode(tok, d)
        n_bytes += len(d.encode("utf-8"))
        n_chars += len(d)
        n_tokens += len(ids)
        if unk is not None:
            n_unk += sum(1 for i in ids if i == unk)
    boundary = len(docs) if has_eos else 0
    return {"docs": len(docs), "bytes": n_bytes + boundary, "chars": n_chars + boundary,
            "tokens": n_tokens + boundary, "unk": n_unk}


def tokenizer_stats(tok: Any, docs: Sequence[str] | str, *,
                    sample: int | None = None) -> dict[str, Any]:
    """Compression and quality statistics of ``tok`` on ``docs``.

    Keys: ``vocab_size``, ``bytes_per_token``, ``chars_per_token``,
    ``tokens_per_doc_mean``, ``unk_rate`` (share of ``unk_id`` tokens),
    ``roundtrip_ok`` (``decode(encode(x)) == x`` on the sample) and
    ``encode_mb_per_s``. ``sample=n`` uses the first ``n`` documents.
    """
    if isinstance(docs, str):
        docs = [docs]
    docs = [d for d in docs if d]
    if sample is not None:
        docs = docs[: max(1, int(sample))]
    t0 = time.perf_counter()
    c = corpus_counts(tok, docs)
    dt = max(time.perf_counter() - t0, 1e-9)
    roundtrip = True
    decode = getattr(tok, "decode", None)
    if callable(decode):
        for d in docs[:64]:
            try:
                if decode(_encode(tok, d)) != d:
                    roundtrip = False
                    break
            except Exception:  # noqa: BLE001
                roundtrip = False
                break
    else:
        roundtrip = None
    tokens = max(1, c["tokens"])
    raw_bytes = sum(len(d.encode("utf-8")) for d in docs)
    return {
        "vocab_size": int(getattr(tok, "vocab_size", 0) or 0),
        "bytes_per_token": c["bytes"] / tokens,
        "chars_per_token": c["chars"] / tokens,
        "tokens_per_doc_mean": c["tokens"] / max(1, c["docs"]),
        "unk_rate": c["unk"] / tokens,
        "roundtrip_ok": roundtrip,
        "encode_mb_per_s": raw_bytes / 1e6 / dt,
    }


__all__ = ["corpus_counts", "tokenizer_stats"]
