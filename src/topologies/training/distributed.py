"""Distributed training support: one launch path, ``torchrun``.

The Trainer never spawns processes or creates the process group. Launch with
``torchrun --nproc_per_node=N script.py`` and call
``torch.distributed.init_process_group(...)`` before constructing the Trainer;
the Trainer then wraps the model in DDP, shards data per rank and reduces
metrics. It never destroys a process group it did not create.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from topologies.training.errors import TrainingError


@dataclass
class DistState:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    strategy: str = "single"

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def resolve_strategy(strategy: str = "auto", devices: object = "auto") -> str:
    s = (strategy or "auto").strip().lower()
    if s in ("single", "ddp"):
        return s
    # auto: single unless multiple devices explicitly requested.
    if isinstance(devices, int) and devices > 1:
        return "ddp"
    if isinstance(devices, (list, tuple)) and len(devices) > 1:
        return "ddp"
    return "single"


def _initialized() -> bool:
    import torch.distributed as dist

    return dist.is_available() and dist.is_initialized()


def get_dist_state() -> DistState:
    if _initialized():
        import torch.distributed as dist

        return DistState(
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            local_rank=int(os.environ.get("LOCAL_RANK", "0")),
            strategy="ddp",
        )
    return DistState()


def wrap_model_ddp(model: nn.Module, device: torch.device) -> nn.Module:
    from torch.nn.parallel import DistributedDataParallel as DDP

    return DDP(model, device_ids=[device.index] if device.type == "cuda" else None)


def maybe_wrap_sampler(loader: Any, *, strategy: str, evaluation: bool = False) -> Any:
    """Shard ``loader`` across ranks when DDP is active.

    Training loaders get a ``DistributedSampler`` that keeps the loader's
    shuffling (a ``RandomSampler`` means shuffle). Evaluation loaders get a
    non-padding ``ShardedEvalSampler``. Loaders that already use a distributed
    sampler and plain iterables are returned unchanged; an ``IterableDataset``
    must shard itself and declare ``sharded = True``.
    """
    if strategy != "ddp" or not _initialized() or not isinstance(loader, DataLoader):
        return loader
    dataset = loader.dataset
    if isinstance(dataset, IterableDataset):
        if getattr(dataset, "sharded", False):
            return loader
        raise TrainingError(
            f"{type(dataset).__name__} is an IterableDataset; under DDP it must yield a "
            "different shard on every rank.", stage="data",
            hint="Shard inside __iter__ (e.g. by torch.distributed.get_rank()) and set the "
            "dataset attribute sharded = True.")
    from topologies.data.samplers import ShardedEvalSampler

    if isinstance(loader.sampler, (DistributedSampler, ShardedEvalSampler)):
        return loader
    state = get_dist_state()
    if evaluation:
        sampler: Any = ShardedEvalSampler(dataset, state.rank, state.world_size)
    else:
        sampler = DistributedSampler(dataset, shuffle=isinstance(loader.sampler, RandomSampler))
    return DataLoader(
        dataset,
        batch_size=loader.batch_size,
        sampler=sampler,
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
        pin_memory=getattr(loader, "pin_memory", False),
        drop_last=loader.drop_last,
    )


def set_sampler_epoch(loader: Any, epoch: int) -> None:
    """Give a distributed sampler its epoch so each epoch reshuffles differently."""
    sampler = getattr(loader, "sampler", None)
    if callable(getattr(sampler, "set_epoch", None)):
        sampler.set_epoch(int(epoch))


def _reduce_device(device: torch.device | str | None) -> torch.device:
    import torch.distributed as dist

    if dist.get_backend() == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device(device) if device is not None else torch.device("cpu")


def all_reduce_mean_dict(values: dict[str, float], state: DistState,
                         device: torch.device | str | None = None) -> dict[str, float]:
    """Mean of each metric across ranks in **one** collective (sorted key order).

    Every rank must pass the same keys. Errors propagate.
    """
    if state.world_size <= 1 or not values:
        return dict(values)
    import torch.distributed as dist

    keys = sorted(values)
    packed = torch.tensor([float(values[k]) for k in keys], dtype=torch.float64,
                          device=_reduce_device(device))
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    packed /= state.world_size
    return {k: float(v) for k, v in zip(keys, packed.tolist())}


def all_reduce_mean(value: float, state: DistState,
                    device: torch.device | str | None = None) -> float:
    """Mean of one scalar across ranks (errors propagate)."""
    return all_reduce_mean_dict({"value": value}, state, device)["value"]


def barrier(state: DistState) -> None:
    """Synchronise ranks (no-op single-process). Errors propagate."""
    if state.world_size > 1 and _initialized():
        import torch.distributed as dist

        dist.barrier()


def cleanup(state: DistState, *, owned: bool = False) -> None:
    """Destroy the process group only when the caller created it (``owned``)."""
    if owned and state.strategy == "ddp" and _initialized():
        import torch.distributed as dist

        dist.destroy_process_group()
