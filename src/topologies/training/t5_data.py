from __future__ import annotations

import itertools
import random
from collections.abc import Sequence

import torch
from torch.utils.data import Dataset


class PretokenizedSpanCorruptor:
    """T5 span corruption over tokenizer-produced integer IDs (never text/bytes)."""

    def __init__(self, *, sentinel_ids: Sequence[int], eos_token_id: int = 1,
                 pad_token_id: int = 0, decoder_start_token_id: int | None = None,
                 noise_density: float = .15, mean_span_length: float = 3., seed: int = 0):
        if not sentinel_ids or len(set(sentinel_ids)) != len(sentinel_ids):
            raise ValueError("sentinel_ids must be a nonempty unique sequence")
        if not 0 < noise_density < 1 or mean_span_length <= 0:
            raise ValueError("noise_density must be in (0,1) and mean_span_length positive")
        self.sentinels = tuple(int(v) for v in sentinel_ids)
        self.pad = int(pad_token_id)
        self.decoder_start = self.pad if decoder_start_token_id is None else int(decoder_start_token_id)
        if min((*self.sentinels, int(eos_token_id), self.pad, self.decoder_start)) < 0:
            raise ValueError("special token IDs must be nonnegative")
        if self.decoder_start != self.pad:
            raise ValueError("T5 decoder_start_token_id must equal PAD")
        if len({*self.sentinels, int(eos_token_id), self.pad}) != len(self.sentinels) + 2:
            raise ValueError("sentinel, EOS, and PAD IDs must be distinct")
        self.eos, self.noise_density, self.mean_span_length = eos_token_id, noise_density, mean_span_length
        self.rng = random.Random(seed)

    def corrupt(self, token_ids: Sequence[int]) -> tuple[list[int], list[int]]:
        ids = [int(i) for i in token_ids]
        if not ids or any(token in {self.eos, self.pad, self.decoder_start, *self.sentinels} for token in ids):
            raise ValueError("input must contain only body IDs (no EOS/PAD/start/sentinels)")
        n = len(ids)
        # Match T5 random_spans_noise_mask: round noise count, reserve at least
        # one unmasked token for nontrivial rows, then randomly segment each
        # partition into approximately equal-count spans.
        noise = min(n, max(1, round(n * self.noise_density)))
        if n > 1:
            noise = min(noise, n - 1)
        nonnoise = n - noise
        spans = max(1, round(noise / self.mean_span_length))
        spans = min(spans, noise, max(1, nonnoise), len(self.sentinels))

        def segmentation(total: int, count: int) -> list[int]:
            # Randomly place count-1 boundaries among total-1 internal slots.
            boundaries = set(self.rng.sample(range(1, total), count - 1))
            points = [0, *sorted(boundaries), total]
            return [b - a for a, b in itertools.pairwise(points)]

        noise_lengths = segmentation(noise, spans)
        keep_lengths = segmentation(nonnoise, spans) if nonnoise else [0] * spans
        source, target, cursor = [], [], 0
        for index, (keep, masked) in enumerate(zip(keep_lengths, noise_lengths)):
            source.extend(ids[cursor:cursor + keep]); cursor += keep
            sentinel = self.sentinels[index]
            source.append(sentinel)
            target.extend((sentinel, *ids[cursor:cursor + masked])); cursor += masked
        source.extend(ids[cursor:]); source.append(self.eos); target.append(self.eos)
        return source, target


class SpanDenoisingCollator:
    def __init__(self, *, pad_token_id: int = 0, decoder_start_token_id: int | None = None, **kwargs):
        self.corruptor = PretokenizedSpanCorruptor(pad_token_id=pad_token_id,
            decoder_start_token_id=decoder_start_token_id, **kwargs)
        self.pad, self.start = self.corruptor.pad, self.corruptor.decoder_start

    def __call__(self, rows: Sequence[Sequence[int]]) -> dict[str, torch.Tensor]:
        if not rows: raise ValueError("batch cannot be empty")
        pairs = [self.corruptor.corrupt(row) for row in rows]
        sw, tw = max(len(p[0]) for p in pairs), max(len(p[1]) for p in pairs)
        src = torch.full((len(rows), sw), self.pad, dtype=torch.long)
        tgt = torch.full((len(rows), tw), self.pad, dtype=torch.long)
        labels = torch.full((len(rows), tw), -100, dtype=torch.long)
        for i, (s, y) in enumerate(pairs):
            src[i, :len(s)] = torch.tensor(s)
            labels[i, :len(y)] = torch.tensor(y)
            tgt[i, 0] = self.start
            if len(y) > 1: tgt[i, 1:len(y)] = torch.tensor(y[:-1])
        return {"src": src, "tgt": tgt, "labels": labels}


class PretokenizedDenoisingDataset(Dataset):
    def __init__(self, rows: Sequence[Sequence[int]]): self.rows = [tuple(map(int, row)) for row in rows]
    def __len__(self): return len(self.rows)
    def __getitem__(self, index): return self.rows[index]
