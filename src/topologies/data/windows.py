"""Virtual epochs over a memory-mapped token stream.

Epoch ``e`` owns windows ``[e * K, (e + 1) * K)`` of the stream. Those windows
do not overlap until the stream is exhausted; after that, indexing wraps.
``shuffle_within_epoch`` permutes the order inside the epoch and leaves the
set of windows unchanged.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import Dataset

from topologies.data.stream import TokenStream


class TokenWindows(Dataset):
    """``samples_per_epoch`` non-overlapping blocks from a ``TokenStream``.

    ``order="sequential"`` cuts the memmap into contiguous blocks of
    ``block_size``. ``set_epoch`` selects which slice of that sequence this
    epoch reads. Items are ``{"tokens": LongTensor[block_size]}``.
    """

    def __init__(
        self,
        stream: TokenStream,
        block_size: int = 512,
        samples_per_epoch: int = 15_625,
        *,
        order: str = "sequential",
        shuffle_within_epoch: bool = True,
        seed: int = 0,
    ) -> None:
        if order != "sequential":
            raise ValueError(f"order must be 'sequential'; got {order!r}.")
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1; got {block_size}.")
        if samples_per_epoch < 1:
            raise ValueError(f"samples_per_epoch must be >= 1; got {samples_per_epoch}.")
        self.stream = stream
        self.block_size = int(block_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.order = order
        self.shuffle_within_epoch = bool(shuffle_within_epoch)
        self.seed = int(seed)
        self.n_windows = int(stream.tokens.shape[0]) // self.block_size
        if self.n_windows < 1:
            raise ValueError(
                f"Stream has {int(stream.tokens.shape[0])} tokens, shorter than "
                f"one block of {self.block_size}."
            )
        self._epoch = 0
        self._perm_epoch: int | None = None
        self._perm: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def window_index(self, index: int) -> int:
        """Stream window owned by ``index`` in the current epoch."""
        local = self._permutation()[int(index)]
        absolute = self._epoch * self.samples_per_epoch + local
        return absolute % self.n_windows

    def token_span(self, index: int) -> tuple[int, int]:
        start = self.window_index(index) * self.block_size
        return start, start + self.block_size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not 0 <= int(index) < self.samples_per_epoch:
            raise IndexError(index)
        start, end = self.token_span(index)
        chunk = np.array(self.stream.tokens[start:end], dtype=np.int64, copy=True)
        return {"tokens": torch.from_numpy(chunk)}

    def _permutation(self) -> list[int]:
        if self._perm_epoch != self._epoch:
            order = list(range(self.samples_per_epoch))
            if self.shuffle_within_epoch:
                random.Random(self.seed + self._epoch * 1_000_003).shuffle(order)
            self._perm = order
            self._perm_epoch = self._epoch
        return self._perm
