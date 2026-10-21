"""Text primitives for dedup, filters and quality metrics (pure Python + torch).

- ``normalize``: NFKC, casefold, punctuation stripped, whitespace collapsed.
- ``shingles``: character (or word) n-gram sets, hashed with ``crc32`` (stable across
  processes, unlike ``hash``).
- ``MinHasher``: vectorized MinHash signatures (torch, ``(a*x + b) mod p`` with
  ``p = 2^31 - 1``) and ``LSHIndex`` (banded, LSM-style sorted runs: memory is ~12 bytes
  per band entry + a 16-bit signature per permutation, so 1M short texts fit in ~0.5 GB).
- ``simhash64`` / ``hamming``: 64-bit SimHash for cheap near-duplicate checks.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
import zlib
from collections.abc import Iterable, Sequence
from typing import Any

import torch

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")
_WORD = re.compile(r"\w+", re.UNICODE)
MERSENNE = (1 << 31) - 1


def normalize(text: str) -> str:
    """NFKC + casefold + punctuation removed + whitespace collapsed."""
    t = unicodedata.normalize("NFKC", str(text)).casefold()
    t = _PUNCT.sub(" ", t)
    return _WS.sub(" ", t).strip()


def words(text: str) -> list[str]:
    return _WORD.findall(str(text).casefold())


def ngrams(seq: Sequence[Any], n: int) -> list[tuple]:
    return [tuple(seq[i:i + n]) for i in range(max(0, len(seq) - n + 1))]


def shingles(text: str, *, k: int = 5, unit: str = "char") -> set[int]:
    """Hashed shingles of the normalized text (``unit="char"`` or ``"word"``)."""
    t = normalize(text)
    if unit == "word":
        w = t.split()
        grams = [" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))] if w else [""]
    else:
        grams = [t[i:i + k] for i in range(max(1, len(t) - k + 1))] if t else [""]
    return {zlib.crc32(g.encode("utf-8", "surrogatepass")) for g in grams}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def text_of(record: dict, key: Any = None) -> str:
    """Text used for dedup / filters: a field, several fields, ``fn(record)``, or every
    non-meta string field (JSON for structured values)."""
    import json

    if callable(key):
        return str(key(record))
    if isinstance(key, str):
        v = record.get(key, (record.get("_attrs") or {}).get(key, ""))
        return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, sort_keys=True)
    if isinstance(key, (list, tuple)):
        return "\n".join(text_of(record, k) for k in key)
    parts = []
    for k, v in record.items():
        if str(k).startswith("_"):
            continue
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, (dict, list)) and not (isinstance(v, list) and v and
                                                  isinstance(v[0], (int, float))):
            parts.append(json.dumps(v, ensure_ascii=False, sort_keys=True))
    return "\n".join(parts)


# ---------------------------------------------------------------- MinHash


def lsh_params(threshold: float, num_perm: int = 128) -> tuple[int, int]:
    """``(bands, rows)`` with ``bands * rows <= num_perm`` minimizing false negatives at
    ``threshold`` (weighted 3:1 over false positives at ``threshold - 0.25``)."""
    best, best_cost = (num_perm // 8, 8), math.inf
    for r in range(1, num_perm + 1):
        b = num_perm // r
        if b < 1:
            break
        fn = 1 - (1 - threshold ** r) ** b
        fn = 1 - fn
        low = max(0.0, threshold - 0.25)
        fp = 1 - (1 - low ** r) ** b
        cost = 3 * fn + fp
        if cost < best_cost:
            best, best_cost = (b, r), cost
    return best


class MinHasher:
    """Vectorized MinHash signatures (``num_perm`` permutations, fixed seed)."""

    def __init__(self, num_perm: int = 128, *, seed: int = 1, k: int = 5,
                 unit: str = "char") -> None:
        g = torch.Generator().manual_seed(seed)
        self.num_perm, self.k, self.unit = int(num_perm), int(k), unit
        self.a = torch.randint(1, MERSENNE, (self.num_perm,), generator=g, dtype=torch.int64)
        self.b = torch.randint(0, MERSENNE, (self.num_perm,), generator=g, dtype=torch.int64)

    def signatures(self, texts: Sequence[str], *, chunk: int = 512) -> torch.Tensor:
        """``[n, num_perm]`` int64 signatures."""
        out = []
        for s in range(0, len(texts), chunk):
            sets = [sorted(shingles(t, k=self.k, unit=self.unit)) for t in texts[s:s + chunk]]
            L = max(len(x) for x in sets) if sets else 1
            # Pad by repeating the first shingle (does not change the min).
            x = torch.tensor([v + [v[0]] * (L - len(v)) for v in sets], dtype=torch.int64)
            x = x & 0xFFFFFFFF
            h = (x.unsqueeze(-1) * self.a + self.b) % MERSENNE  # [n, L, P]
            out.append(h.min(dim=1).values)
        return torch.cat(out) if out else torch.empty(0, self.num_perm, dtype=torch.int64)

    @staticmethod
    def similarity(sig_a: torch.Tensor, sig_b: torch.Tensor) -> torch.Tensor:
        """Estimated Jaccard (fraction of equal signature entries)."""
        return (sig_a == sig_b).float().mean(dim=-1)


class LSHIndex:
    """Banded LSH over MinHash signatures with bounded, linear memory.

    Band keys live in sorted int64 runs merged LSM-style (lookups are
    ``searchsorted``); verification uses 16-bit signature fingerprints
    (b-bit MinHash), corrected for chance collisions.
    """

    def __init__(self, num_perm: int = 128, threshold: float = 0.85) -> None:
        self.num_perm = int(num_perm)
        self.threshold = float(threshold)
        self.bands, self.rows = lsh_params(threshold, num_perm)
        self._runs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._recent: dict[int, int] = {}
        self._sigs: list[torch.Tensor] = []
        self._sig_cat: torch.Tensor | None = None
        self.size = 0
        self.flush_at = 1 << 15
        g = torch.Generator().manual_seed(7)
        self._mix = torch.randint(1, 1 << 30, (self.rows,), generator=g, dtype=torch.int64)

    def band_keys(self, sigs: torch.Tensor) -> torch.Tensor:
        """``[n, bands]`` int64 keys (band index folded in)."""
        n = sigs.shape[0]
        x = sigs[:, : self.bands * self.rows].reshape(n, self.bands, self.rows)
        keys = (x * self._mix).sum(-1)  # wrapping int64 arithmetic
        band = torch.arange(self.bands, dtype=torch.int64) * 1_000_003
        return keys * 31 + band

    def _bits(self, sigs: torch.Tensor) -> torch.Tensor:
        # Low 16 bits stored as int16 (wrapping): equality is all verification needs.
        return ((sigs & 0xFFFF) - 32768).to(torch.int16)

    def _sig_matrix(self) -> torch.Tensor:
        if self._sig_cat is None or self._sig_cat.shape[0] != self.size:
            self._sig_cat = torch.cat(self._sigs) if self._sigs else torch.empty(
                0, self.num_perm, dtype=torch.int16)
            self._sigs = [self._sig_cat]
        return self._sig_cat

    def candidates(self, keys: torch.Tensor) -> list[set[int]]:
        """Stored slots sharing at least one band with each query row."""
        n = keys.shape[0]
        out: list[set[int]] = [set() for _ in range(n)]
        flat = keys.reshape(-1)
        for run_keys, run_slots in self._runs:
            pos = torch.searchsorted(run_keys, flat)
            pos_c = pos.clamp(max=max(0, run_keys.numel() - 1))
            hit = (pos < run_keys.numel()) & (run_keys[pos_c] == flat)
            for q in torch.nonzero(hit).flatten().tolist():
                out[q // self.bands].add(int(run_slots[pos_c[q]]))
        if self._recent:
            for q, k in enumerate(flat.tolist()):
                s = self._recent.get(k)
                if s is not None:
                    out[q // self.bands].add(s)
        return out

    def best_match(self, sig: torch.Tensor, cands: Iterable[int]) -> tuple[int | None, float]:
        cands = list(cands)
        if not cands:
            return None, 0.0
        mat = self._sig_matrix()[torch.tensor(cands)]
        eq = (mat == self._bits(sig.unsqueeze(0))).float().mean(-1)
        # Correct for 16-bit chance collisions.
        est = ((eq - 2 ** -16) / (1 - 2 ** -16)).clamp(min=0.0)
        i = int(est.argmax())
        return cands[i], float(est[i])

    def add(self, sig: torch.Tensor, keys: torch.Tensor, *, flush: bool = True) -> int:
        slot = self.size
        self._sigs.append(self._bits(sig.unsqueeze(0)))
        self.size += 1
        for k in keys.tolist():
            self._recent.setdefault(k, slot)
        if flush:
            self.maybe_flush()
        return slot

    def maybe_flush(self) -> None:
        if len(self._recent) >= self.flush_at:
            self._flush()

    def _flush(self) -> None:
        if not self._recent:
            return
        ks = torch.tensor(list(self._recent.keys()), dtype=torch.int64)
        vs = torch.tensor(list(self._recent.values()), dtype=torch.int64)
        order = torch.argsort(ks)
        self._runs.append((ks[order], vs[order]))
        self._recent = {}
        while len(self._runs) >= 2 and self._runs[-2][0].numel() <= 2 * self._runs[-1][0].numel():
            (k1, v1), (k2, v2) = self._runs[-2], self._runs[-1]
            # Older entries win (first occurrence is the canonical record).
            k = torch.cat([k1, k2])
            v = torch.cat([v1, v2])
            order = torch.argsort(k, stable=True)
            k, v = k[order], v[order]
            keep = torch.ones_like(k, dtype=torch.bool)
            keep[1:] = k[1:] != k[:-1]
            self._runs[-2:] = [(k[keep], v[keep])]

    def memory_bytes(self) -> int:
        runs = sum(k.numel() * 8 + v.numel() * 8 for k, v in self._runs)
        return runs + self.size * self.num_perm * 2 + len(self._recent) * 100


# ---------------------------------------------------------------- SimHash


def simhash64(text: str, *, k: int = 3) -> int:
    """64-bit SimHash over word ``k``-grams (character 4-grams for short texts)."""
    w = normalize(text).split()
    grams = [" ".join(w[i:i + k]) for i in range(len(w) - k + 1)] if len(w) >= k + 2 else \
        [normalize(text)[i:i + 4] for i in range(max(1, len(normalize(text)) - 3))]
    acc = [0] * 64
    for g in grams:
        h = int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "little")
        for bit in range(64):
            acc[bit] += 1 if (h >> bit) & 1 else -1
    out = 0
    for bit in range(64):
        if acc[bit] > 0:
            out |= 1 << bit
    return out


_POP16 = None


def hamming(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Bitwise Hamming distance of int64 tensors (broadcasting)."""
    global _POP16
    if _POP16 is None:
        t = torch.arange(1 << 16, dtype=torch.int64)
        pop = torch.zeros(1 << 16, dtype=torch.int64)
        for s in range(16):
            pop += (t >> s) & 1
        _POP16 = pop
    x = a ^ b
    out = torch.zeros_like(x)
    for s in (0, 16, 32, 48):
        out += _POP16[(x >> s) & 0xFFFF]
    return out


def to_signed64(v: int) -> int:
    return v - (1 << 64) if v >= 1 << 63 else v


__all__ = ["LSHIndex", "MinHasher", "hamming", "jaccard", "lsh_params", "ngrams", "normalize",
           "shingles", "simhash64", "text_of", "to_signed64", "words"]
