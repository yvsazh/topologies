"""Text features for dedup, diversity and fidelity: hashed n-gram vectors.

``get_embedder(None)`` / ``"hashed"``: hashed word uni/bi-gram + char trigram
TF vectors (2^14 dims, L2-normalized) — deterministic, offline, model-free.
Any object with ``encode(texts) -> [n, d]`` works too (e.g. your own feature
function wrapped in a class).
"""

from __future__ import annotations

import itertools
import math
import zlib
from collections.abc import Sequence
from typing import Any

import torch

from topologies.synth.ctor import Recorded
from topologies.synth.textutil import normalize


class HashedEmbedder(Recorded):
    """Hashed TF features (see module docstring); ``idf=True`` re-weights by a corpus."""

    name = "hashed"

    def __init__(self, dim: int = 1 << 14) -> None:
        self.dim = int(dim)
        self.idf: torch.Tensor | None = None

    def _features(self, text: str) -> dict[int, float]:
        t = normalize(text)
        w = t.split()
        feats: dict[int, float] = {}
        grams = list(w) + [f"{a} {b}" for a, b in itertools.pairwise(w)]
        grams += [f"#{t[i:i + 3]}" for i in range(max(0, len(t) - 2))]
        for g in grams:
            h = zlib.crc32(g.encode("utf-8", "surrogatepass")) % self.dim
            feats[h] = feats.get(h, 0.0) + 1.0
        return {k: 1.0 + math.log(v) for k, v in feats.items()}

    def fit_idf(self, texts: Sequence[str]) -> HashedEmbedder:
        df = torch.zeros(self.dim)
        for t in texts:
            for k in self._features(t):
                df[k] += 1
        self.idf = torch.log((1 + len(texts)) / (1 + df)) + 1
        return self

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        rows, cols, vals = [], [], []
        for i, t in enumerate(texts):
            for k, v in self._features(t).items():
                rows.append(i)
                cols.append(k)
                vals.append(v)
        out = torch.zeros(len(texts), self.dim)
        if rows:
            out[torch.tensor(rows), torch.tensor(cols)] = torch.tensor(vals)
        if self.idf is not None:
            out = out * self.idf
        return torch.nn.functional.normalize(out, dim=-1)


_CACHE: dict[str, Any] = {}


def get_embedder(spec: Any = None) -> Any:
    """Embedder for ``spec`` (see module docstring)."""
    if spec is not None and not isinstance(spec, str) and hasattr(spec, "encode"):
        return spec
    key = str(spec or "hashed")
    if key in _CACHE:
        return _CACHE[key]
    if key in ("hashed", "auto"):
        emb: Any = HashedEmbedder()
    else:
        from topologies.synth.errors import unknown

        raise unknown("text featurizer", key, ["hashed"], stage="embed",
                      extra="Or pass an object with encode(texts) -> [n, d].")
    _CACHE[key] = emb
    return emb


__all__ = ["HashedEmbedder", "get_embedder"]
