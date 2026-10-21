from __future__ import annotations

import math
import random
from collections.abc import Sequence

import torch
from torch.utils.data import Dataset


class PretokenizedBARTCorruptor:
    def __init__(self, *, mask_token_id: int, bos_token_id: int = 0, eos_token_id: int = 2,
                 pad_token_id: int = 1, mask_ratio: float = .3, poisson_lambda: float = 3.0,
                 full_stop_id: int | None = None, permute_sentences: float = 1.0, seed: int = 0):
        if not 0 <= mask_ratio <= 1 or poisson_lambda <= 0 or not 0 <= permute_sentences <= 1:
            raise ValueError("invalid denoising parameters")
        self.mask, self.bos, self.eos, self.pad = mask_token_id, bos_token_id, eos_token_id, pad_token_id
        self.mask_ratio, self.poisson_lambda = mask_ratio, poisson_lambda
        self.full_stop_id, self.permute_sentences = full_stop_id, permute_sentences
        self.rng = random.Random(seed)
        self.generator = torch.Generator().manual_seed(seed)

    def permute(self, document: Sequence[int], *, full_stop_id: int | None = None):
        ids = list(map(int, document))
        if len(ids) < 2 or ids[0] != self.bos or ids[-1] != self.eos:
            raise ValueError("document must be [BOS, body..., protected end]")
        stop = self.full_stop_id if full_stop_id is None else full_stop_id
        if stop is None: raise ValueError("full_stop_id is required for sentence permutation")
        body = ids[1:-1]
        segments, start = [], 0
        for i, token in enumerate(body):
            if token == stop:
                segments.append(body[start:i + 1]); start = i + 1
        if start < len(body): segments.append(body[start:])
        if len(segments) > 1:
            count = min(len(segments), math.ceil(len(segments) * self.permute_sentences))
            indexes = list(range(len(segments)))
            selected = indexes[:count]
            reordered = selected[:]
            self.rng.shuffle(reordered)
            for old, new in zip(selected, reordered): indexes[old] = new
            segments = [segments[i] for i in indexes]
        return [self.bos, *(token for segment in segments for token in segment), ids[-1]]

    def _poisson_lengths(self, budget):
        probabilities = torch.tensor([math.exp(-self.poisson_lambda) * self.poisson_lambda**k / math.factorial(k) for k in range(32)])
        lengths, total = [], 0
        while total < budget:
            length = int(torch.multinomial(probabilities, 1, generator=self.generator))
            lengths.append(length)
            total += length
        index = next(i for i, value in enumerate(lengths) if sum(lengths[:i + 1]) >= budget)
        lengths[index] = budget - sum(lengths[:index])
        return lengths[:index + 1]

    def mask_spans(self, document: Sequence[int], *, lengths: Sequence[int] | None = None,
                   starts: Sequence[int] | None = None):
        ids = list(map(int, document))
        if len(ids) < 2 or ids[0] != self.bos or ids[-1] != self.eos:
            raise ValueError("document must be [BOS, body..., protected end]")
        original = ids[1:-1]
        budget = math.ceil(len(original) * self.mask_ratio)
        if lengths is None:
            if budget <= 0: return ids
            lengths = self._poisson_lengths(budget)
        lengths = [int(length) for length in lengths]
        if any(length < 0 for length in lengths): raise ValueError("span lengths must be nonnegative")
        positive = [length for length in lengths if length > 0]
        if starts is None:
            starts = self.rng.sample(range(len(original)), min(len(positive), len(original))) if positive else []
        starts = [int(start) for start in starts]
        if len(starts) != len(positive) or any(start < 0 or start >= len(original) for start in starts):
            raise ValueError("one valid body start is required per positive span")
        masked = [False] * len(original)
        span_start = {}
        for length, start in zip(positive, starts):
            if masked[start]: continue
            span_start[start] = True
            for position in range(start, min(len(original), start + length)):
                masked[position] = True
        output, index = [], 0
        while index < len(original):
            if index in span_start:
                output.append(self.mask)
                while index < len(original) and masked[index]: index += 1
            elif masked[index]:
                index += 1
            else:
                output.append(original[index]); index += 1
        for length in lengths:
            if length == 0:
                output.insert(self.rng.randrange(len(output) + 1), self.mask)
        return [ids[0], *output, ids[-1]]

    def corrupt(self, document: Sequence[int], *, full_stop_id: int | None = None):
        permuted = self.permute(document, full_stop_id=full_stop_id) if self.permute_sentences else list(map(int, document))
        return self.mask_spans(permuted), list(map(int, document))


class BARTDenoisingCollator:
    def __init__(self, *, pad_token_id=1, mask_token_id: int, bos_token_id=0, eos_token_id=2,
                 full_stop_id: int | None = None, permute_sentences: float = 1.0, seed=0, **kwargs):
        self.corruptor = PretokenizedBARTCorruptor(mask_token_id=mask_token_id, bos_token_id=bos_token_id,
            eos_token_id=eos_token_id, pad_token_id=pad_token_id, full_stop_id=full_stop_id,
            permute_sentences=permute_sentences, seed=seed, **kwargs)
        self.pad = pad_token_id

    def __call__(self, rows):
        pairs = [self.corruptor.corrupt(row) for row in rows]
        sw, tw = max(len(s) for s, _ in pairs), max(len(t) for _, t in pairs)
        src = torch.full((len(rows), sw), self.pad, dtype=torch.long)
        tgt = torch.full((len(rows), tw), self.pad, dtype=torch.long)
        labels = torch.full_like(tgt, -100)
        for i, (source, clean) in enumerate(pairs):
            src[i, :len(source)] = torch.tensor(source)
            labels[i, :len(clean)] = torch.tensor(clean)
            tgt[i, :len(clean)] = torch.tensor([clean[-1], *clean[:-1]])
        return {"src": src, "tgt": tgt, "labels": labels}


class PretokenizedBARTDataset(Dataset):
    def __init__(self, rows): self.rows = [tuple(map(int, row)) for row in rows]
    def __len__(self): return len(self.rows)
    def __getitem__(self, index): return self.rows[index]
