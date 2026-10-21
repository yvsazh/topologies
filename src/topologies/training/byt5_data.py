from __future__ import annotations

import random
from collections.abc import Sequence

import torch

PAD_ID, EOS_ID, UNK_ID, BYTE_OFFSET = 0, 1, 2, 3
SENTINEL_COUNT, VOCAB_SIZE = 125, 384


def encode_bytes(text: str, append_eos: bool = True) -> list[int]:
    encoded = [BYTE_OFFSET + byte for byte in text.encode("utf-8")]
    return encoded + ([EOS_ID] if append_eos else [])


def decode_bytes(ids: Sequence[int]) -> str:
    output = bytearray()
    for token in ids:
        if token == EOS_ID:
            break
        if BYTE_OFFSET <= token < BYTE_OFFSET + 256:
            output.append(token - BYTE_OFFSET)
    return output.decode("utf-8", errors="replace")


class ByteSpanDenoising:
    """T5-style random byte-span corruption with descending sentinel tokens."""

    def __init__(self, noise_density: float = 0.15, mean_span_bytes: float = 3.0,
                 sentinels: int = SENTINEL_COUNT, seed: int | None = None) -> None:
        if not 0 < noise_density < 1 or mean_span_bytes <= 0:
            raise ValueError("noise_density must be in (0, 1) and mean_span_bytes positive")
        if not 1 <= sentinels <= SENTINEL_COUNT:
            raise ValueError("sentinels must be between 1 and 125")
        self.noise_density, self.mean_span_bytes, self.sentinels = noise_density, mean_span_bytes, sentinels
        self.rng = random.Random(seed)

    def corrupt(self, text: str) -> tuple[list[int], list[int]]:
        data = list(text.encode("utf-8"))
        if not data:
            return [EOS_ID], [EOS_ID]
        noise_count = max(1, min(len(data), round(len(data) * self.noise_density)))
        nonnoise_count = len(data) - noise_count
        span_count = min(max(1, round(noise_count / self.mean_span_bytes)), noise_count,
                         max(1, nonnoise_count))
        if span_count > self.sentinels:
            raise ValueError("corruption requires more sentinel tokens than configured")
        noise_lengths = self._random_partition(noise_count, span_count)
        nonnoise_lengths = self._random_partition(nonnoise_count, span_count)
        segments = []
        for index, noise_length in enumerate(noise_lengths):
            segments.extend((nonnoise_lengths[index], noise_length))
        source, target = [], []
        cursor = 0
        for span_index, (nonnoise_length, noise_length) in enumerate(zip(segments[::2], segments[1::2])):
            source.extend(BYTE_OFFSET + value for value in data[cursor:cursor + nonnoise_length])
            cursor += nonnoise_length
            sentinel = 258 + self.sentinels - span_index
            source.append(sentinel)
            target.append(sentinel)
            target.extend(BYTE_OFFSET + value for value in data[cursor:cursor + noise_length])
            cursor += noise_length
        source.extend(BYTE_OFFSET + value for value in data[cursor:])
        source.append(EOS_ID)
        target.append(EOS_ID)
        return source, target

    def _random_partition(self, total: int, parts: int) -> list[int]:
        if total == 0:
            return [0] * parts
        if parts <= 0 or total < parts:
            raise ValueError("cannot partition bytes into the requested positive spans")
        boundaries = sorted(self.rng.sample(range(1, total), parts - 1)) if parts > 1 else []
        endpoints = [0, *boundaries, total]
        return [endpoints[index + 1] - endpoints[index] for index in range(parts)]


class ByteSpanDenoisingCollator:
    def __init__(self, noise_density: float = 0.15, mean_span_bytes: float = 3.0,
                 seed: int | None = None) -> None:
        self.corruptor = ByteSpanDenoising(noise_density, mean_span_bytes, seed=seed)

    def __call__(self, batch: Sequence[str]) -> dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("batch must not be empty")
        pairs = [self.corruptor.corrupt(text) for text in batch]
        src_width = max(len(source) for source, _ in pairs)
        tgt_width = max(len(target) for _, target in pairs)
        src = torch.full((len(batch), src_width), PAD_ID, dtype=torch.long)
        tgt = torch.full((len(batch), tgt_width), PAD_ID, dtype=torch.long)
        labels = torch.full((len(batch), tgt_width), -100, dtype=torch.long)
        for index, (source, target) in enumerate(pairs):
            src[index, :len(source)] = torch.tensor(source)
            labels[index, :len(target)] = torch.tensor(target)
            tgt[index, 0] = PAD_ID
            if len(target) > 1:
                tgt[index, 1:len(target)] = torch.tensor(target[:-1])
        return {"src": src, "tgt": tgt, "labels": labels}
