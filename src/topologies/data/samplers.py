"""Samplers for distributed evaluation."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from torch.utils.data import Sampler


class ShardedEvalSampler(Sampler[int]):
    """Strided, **non-padding** shard of a map-style dataset for one rank.

    Rank ``r`` of ``world_size`` gets indices ``r, r + world_size, ...``: every
    sample is seen by exactly one rank and none is duplicated, so weighted sums
    reduced across ranks are exact (``DistributedSampler`` pads instead).
    Shards may differ in length by one; evaluation must reduce sums, not means.
    """

    nonpadding = True

    def __init__(self, dataset: Any, rank: int, world_size: int) -> None:
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(f"invalid rank {rank} for world_size {world_size}")
        self.dataset, self.rank, self.world_size = dataset, int(rank), int(world_size)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        n = len(self.dataset)
        return max(0, (n - self.rank + self.world_size - 1) // self.world_size)
