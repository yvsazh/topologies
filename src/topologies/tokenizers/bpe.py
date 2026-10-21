"""Trainable byte-level BPE tokenizer (pure Python, no dependencies).

Byte-level means the base alphabet is the 256 byte values, so *any* text
encodes losslessly without an unknown token. Training learns merges over
pre-tokenized words (GPT-2 style splitting) with incremental pair counts,
which keeps vocabularies of a few thousand tokens fast on CPU.
"""

from __future__ import annotations

import heapq
import re
from collections import Counter, defaultdict
from itertools import pairwise
from typing import Any

from topologies.tokenizers.base import Tokenizer, iter_texts

# GPT-2 pre-tokenization with stdlib ``re``: letters, digits, other symbols
# (an optional leading space stays attached to the following word).
PRETOKENIZE_RE = re.compile(
    r"'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+",
    re.UNICODE,
)
_CACHE_LIMIT = 200_000


def _learn_merges(
    word_counts: dict[bytes, int], num_merges: int, min_frequency: int
) -> list[tuple[int, int]]:
    words: list[list[int]] = [list(w) for w in word_counts]
    freqs: list[int] = list(word_counts.values())
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    where: dict[tuple[int, int], set[int]] = defaultdict(set)
    for wi, w in enumerate(words):
        f = freqs[wi]
        for p in pairwise(w):
            pair_counts[p] += f
            where[p].add(wi)
    heap = [(-c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)
    merges: list[tuple[int, int]] = []
    next_id = 256
    while len(merges) < num_merges and heap:
        negc, pair = heapq.heappop(heap)
        current = pair_counts.get(pair, 0)
        if current != -negc:
            if current > 0:
                heapq.heappush(heap, (-current, pair))
            continue
        if current < min_frequency:
            break
        a, b = pair
        new = next_id
        next_id += 1
        merges.append(pair)
        touched: set[tuple[int, int]] = set()
        for wi in where.pop(pair, ()):
            w = words[wi]
            if len(w) < 2:
                continue
            f = freqs[wi]
            for p in pairwise(w):
                pair_counts[p] -= f
                touched.add(p)
            merged: list[int] = []
            i = 0
            n = len(w)
            while i < n:
                if i < n - 1 and w[i] == a and w[i + 1] == b:
                    merged.append(new)
                    i += 2
                else:
                    merged.append(w[i])
                    i += 1
            words[wi] = merged
            for p in pairwise(merged):
                pair_counts[p] += f
                where[p].add(wi)
                touched.add(p)
        pair_counts.pop(pair, None)
        for p in touched:
            c = pair_counts.get(p, 0)
            if c > 0:
                heapq.heappush(heap, (-c, p))
            elif p in pair_counts:
                del pair_counts[p]
    return merges


class BPETokenizer(Tokenizer):
    """Byte-level BPE. Train once, save, and reuse with any token model.

    >>> tok = BPETokenizer.train("corpus.txt", vocab_size=2048)
    >>> tok.save("artifacts/tokenizer")
    >>> model = TransformerLM(vocab_size=tok.vocab_size, ...)
    """

    kind = "bpe"
    needs_unk = False

    def __init__(self, merges: list[tuple[int, int]] = (), **special_kw: Any) -> None:
        super().__init__(**special_kw)
        self.merges = [(int(a), int(b)) for a, b in merges]
        self._ranks = {pair: r for r, pair in enumerate(self.merges)}
        self._bytes: list[bytes] = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self._bytes.append(self._bytes[a] + self._bytes[b])
        self._cache: dict[str, list[int]] = {}

    @classmethod
    def train(
        cls,
        source: Any,
        vocab_size: int = 4096,
        *,
        min_frequency: int = 2,
        **special_kw: Any,
    ) -> BPETokenizer:
        """Learn merges until ``vocab_size`` (specials + 256 bytes + merges)."""
        probe = cls(**special_kw)
        base = probe.num_special_tokens + 256
        if vocab_size < base:
            raise ValueError(
                f"vocab_size={vocab_size} is below the byte-level minimum {base} "
                f"(256 bytes + {probe.num_special_tokens} special tokens)."
            )
        counts: Counter[bytes] = Counter()
        specials = probe.all_special_tokens
        for text in iter_texts(source):
            for s in specials:
                text = text.replace(s, "\n")
            counts.update(m.encode("utf-8") for m in PRETOKENIZE_RE.findall(text))
        merges = _learn_merges(dict(counts), vocab_size - base, max(1, int(min_frequency)))
        return cls(merges, **special_kw)

    def _encode_word(self, word: str) -> list[int]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        ids = list(word.encode("utf-8"))
        ranks = self._ranks
        while len(ids) > 1:
            best = None
            best_rank = None
            for p in pairwise(ids):
                r = ranks.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = p, r
            if best is None:
                break
            new = 256 + best_rank  # type: ignore[operator]
            a, b = best
            out: list[int] = []
            i = 0
            n = len(ids)
            while i < n:
                if i < n - 1 and ids[i] == a and ids[i + 1] == b:
                    out.append(new)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        if len(self._cache) < _CACHE_LIMIT:
            self._cache[word] = ids
        return ids

    def _encode_plain(self, text: str) -> list[int]:
        out: list[int] = []
        for word in PRETOKENIZE_RE.findall(text):
            out.extend(self._encode_word(word))
        return out

    def _decode_plain(self, ids: list[int]) -> str:
        return b"".join(self._bytes[i] for i in ids).decode("utf-8", errors="replace")

    def _content_token(self, cid: int) -> str:
        return self._bytes[cid].decode("utf-8", errors="backslashreplace")

    def _content_size(self) -> int:
        return 256 + len(self.merges)

    def _state(self) -> dict:
        return {"merges": [list(m) for m in self.merges]}

    @classmethod
    def _from_state(cls, state: dict, **kw: Any) -> BPETokenizer:
        kw.pop("_directory", None)
        return cls([tuple(m) for m in state.get("merges", [])], **kw)
