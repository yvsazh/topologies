"""Privacy checks against private real records (e.g. a reference sample a generator
was calibrated on): generated data must not reproduce them.

- distance to closest real record (cosine on hashed n-gram features) for every
  synthetic record, compared with a holdout baseline (held-out real records ->
  remaining real records): synthetic records much closer than real records are to
  each other are suspicious;
- exact memorization: a shared substring of ``min_substring`` (50) characters;
- a flagged list (memorized, or closer than the baseline's 1st percentile).
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from typing import Any

import torch

from topologies.synth.textutil import text_of


def _windows(text: str, n: int) -> set[int]:
    t = re.sub(r"\s+", " ", text.strip().casefold())
    return {hash(t[i:i + n]) for i in range(max(0, len(t) - n + 1))}


def privacy(synthetic: Sequence[dict], seeds: Sequence[dict], *, key: Any = None,
            min_substring: int = 50, holdout: float = 0.3, embed: Any = None,
            seed: int = 0, max_records: int = 5000) -> dict[str, Any]:
    from topologies.synth.quality.embed import get_embedder

    rng = random.Random(seed)
    syn = list(synthetic)
    if len(syn) > max_records:
        syn = rng.sample(syn, max_records)
    sd = list(seeds)
    if not sd or not syn:
        return {"note": "no seeds or no records"}
    e = get_embedder(embed)
    s_txt = [text_of(r, key) for r in syn]
    d_txt = [text_of(r, key) for r in sd]
    xs = torch.nn.functional.normalize(torch.as_tensor(e.encode(s_txt), dtype=torch.float32),
                                       dim=-1)
    xd = torch.nn.functional.normalize(torch.as_tensor(e.encode(d_txt), dtype=torch.float32),
                                       dim=-1)
    nearest = (xs @ xd.T).max(1)
    sims = nearest.values.tolist()
    base: list[float] = []
    if len(sd) >= 4:
        idx = list(range(len(sd)))
        rng.shuffle(idx)
        k = max(1, int(len(sd) * holdout))
        ho, tr = idx[:k], idx[k:]
        base = (xd[ho] @ xd[tr].T).max(1).values.tolist()
    seed_windows: set[int] = set()
    for t in d_txt:
        seed_windows |= _windows(t, min_substring)
    memorized = []
    for i, t in enumerate(s_txt):
        if len(t) >= min_substring and _windows(t, min_substring) & seed_windows:
            memorized.append(i)
    thr = sorted(base)[max(0, int(0.99 * len(base)) - 1)] if base else 0.98
    flagged = sorted(set(memorized) | {i for i, s in enumerate(sims) if s > max(thr, 0.9)})

    def stats(v: list[float]) -> dict[str, float]:
        if not v:
            return {}
        s = sorted(v)
        return {"mean": sum(s) / len(s), "p50": s[len(s) // 2], "p95": s[int(0.95 * (len(s) -
                                                                                      1))],
                "max": s[-1]}

    return {"n": len(syn), "seeds": len(sd), "nearest_similarity": stats(sims),
            "baseline_similarity": stats(base),
            "closer_than_baseline_p99": sum(1 for s in sims if s > thr) / len(sims),
            "memorized": len(memorized),
            "flagged": [{**syn[i], "_nearest_similarity": round(sims[i], 4),
                         "_nearest_seed": d_txt[int(nearest.indices[i])][:200]}
                        for i in flagged[:50]],
            "flagged_count": len(flagged)}


__all__ = ["privacy"]
