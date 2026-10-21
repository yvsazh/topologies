"""Tokenizers: an independent axis, interoperable with any token-based model.

Train (or import) a tokenizer once, save it, and pass ``tokenizer.vocab_size``
to whichever model you build:

    from topologies.tokenizers import train_tokenizer, load_tokenizer

    tok = train_tokenizer("corpus.txt", kind="bpe", vocab_size=2048)
    tok.save("artifacts/tokenizer")
    tok = load_tokenizer("artifacts/tokenizer")

Built-ins: ``CharTokenizer`` (dev/smoke), ``WordTokenizer``, trainable
byte-level ``BPETokenizer`` (pure Python), and ``HFTokenizer`` wrapping the
optional ``tokenizers`` package (``pip install 'topologies[nlp]'``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from topologies.tokenizers.base import (
    DEFAULT_SPECIAL_TOKENS,
    TOKENIZER_FILE,
    Tokenizer,
    tokenizer_types,
)
from topologies.tokenizers.bpe import BPETokenizer
from topologies.tokenizers.hf import HFTokenizer
from topologies.tokenizers.simple import CharTokenizer, WordTokenizer
from topologies.tokenizers.stats import corpus_counts, tokenizer_stats

_KINDS = {"char": CharTokenizer, "word": WordTokenizer, "bpe": BPETokenizer, "hf": HFTokenizer}


def train_tokenizer(source: Any, kind: str = "bpe", vocab_size: int | None = None,
                    **kwargs: Any) -> Tokenizer:
    """Train a tokenizer on text.

    ``source``: raw text, a file path, or an iterable of texts/paths.
    ``kind``: ``"char"`` | ``"word"`` | ``"bpe"`` | ``"hf"`` (Rust backend).
    ``vocab_size``: target size including special tokens (ignored by ``char``).
    Extra keyword arguments go to the tokenizer's ``train`` (e.g.
    ``special_tokens={"eos": "</s>"}``, ``min_frequency=2``).
    """
    k = str(kind).strip().lower()
    cls = _KINDS.get(k)
    if cls is None:
        raise ValueError(f"Unknown tokenizer kind {kind!r}; available: {sorted(_KINDS)}.")
    if k == "char":
        if vocab_size is not None:
            raise ValueError("CharTokenizer derives its vocabulary from the text; "
                             "omit vocab_size or use kind='bpe'.")
        return cls.train(source, **kwargs)
    if vocab_size is None:
        vocab_size = 4096 if k in ("bpe", "hf") else None
    return cls.train(source, vocab_size, **kwargs)


def load_tokenizer(path: str | Path) -> Tokenizer:
    """Load any tokenizer saved with ``tokenizer.save(...)``."""
    return Tokenizer.load(path)


__all__ = [
    "DEFAULT_SPECIAL_TOKENS",
    "TOKENIZER_FILE",
    "BPETokenizer",
    "CharTokenizer",
    "HFTokenizer",
    "Tokenizer",
    "WordTokenizer",
    "corpus_counts",
    "load_tokenizer",
    "tokenizer_stats",
    "tokenizer_types",
    "train_tokenizer",
]
