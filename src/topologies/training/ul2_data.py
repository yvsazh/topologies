from __future__ import annotations

import random
from collections.abc import Sequence

import torch


class UL2DenoisingCollator:
    mode_tokens = {"R": 0, "S": 1, "X": 2}

    def __init__(self, *, mode="R", sentinel_ids, eos_token_id=1, pad_token_id=0, seed=0):
        if mode not in self.mode_tokens: raise ValueError("mode must be R, S, or X")
        sentinels = tuple(map(int, sentinel_ids))
        if len(sentinels) < 4: raise ValueError("three mode tokens and at least one span sentinel are required")
        self.mode, self.mode_token_id, self.sentinels = mode, sentinels["RSX".index(mode)], sentinels[3:]
        self.eos, self.pad, self.rng = eos_token_id, pad_token_id, random.Random(seed)

    def corrupt(self, row: Sequence[int]):
        ids = list(map(int, row))
        if not ids or any(t in {self.pad, self.eos, *self.sentinels} for t in ids): raise ValueError("rows must contain pretokenized body IDs only")
        n = len(ids)
        if self.mode == "X":
            cut = max(1, min(n - 1, round(n * .75)))
            source, removed = ids[:cut], ids[cut:]
            if not removed: removed, source = source[-1:], source[:-1]
            target = removed
        else:
            choices = ((3, .15), (8, .15)) if self.mode == "R" else ((64, .5),)
            mean, rate = self.rng.choice(choices)
            noise = min(n - 1 if n > 1 else 1, max(1, round(n * rate)))
            span_count = min(len(self.sentinels), max(1, round(noise / mean)))
            starts = sorted(self.rng.sample(range(n), span_count))
            masked = set()
            remaining = noise
            for start in starts:
                take = min(remaining, max(1, round(self.rng.gauss(mean, max(1, mean / 2)))))
                masked.update(range(start, min(n, start + take)))
                remaining = noise - len(masked)
            if remaining:
                for i in range(n):
                    if i not in masked: masked.add(i); remaining -= 1
                    if not remaining: break
            source, target, i, span = [], [], 0, 0
            while i < n:
                if i in masked:
                    sentinel = self.sentinels[span % len(self.sentinels)]; source.append(sentinel); target.append(sentinel)
                    while i < n and i in masked: target.append(ids[i]); i += 1
                    span += 1
                else: source.append(ids[i]); i += 1
        return [self.mode_token_id, *source, self.eos], [*target, self.eos]

    def __call__(self, rows):
        pairs = [self.corrupt(row) for row in rows]
        sw, tw = max(len(x[0]) for x in pairs), max(len(x[1]) for x in pairs)
        src = torch.full((len(rows), sw), self.pad, dtype=torch.long)
        tgt = torch.full((len(rows), tw), self.pad, dtype=torch.long)
        labels = torch.full_like(tgt, -100)
        for i, (s, y) in enumerate(pairs):
            src[i, :len(s)] = torch.tensor(s); labels[i, :len(y)] = torch.tensor(y)
            tgt[i, :len(y)] = torch.tensor([self.pad, *y[:-1]])
        return {"src": src, "tgt": tgt, "labels": labels, "mode": self.mode}
