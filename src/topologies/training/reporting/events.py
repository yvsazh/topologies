"""Immutable reporting events (facts produced by Trainer)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RunInfo:
    model_name: str | None = None
    architecture_id: str | None = None
    param_total: int | None = None
    param_trainable: int | None = None
    param_frozen: int | None = None
    recipe_name: str | None = None
    recipe_level: str | None = None
    device: str | None = None
    precision: str | None = None
    optimizer_name: str | None = None
    lr: float | None = None
    weight_decay: float | None = None
    scheduler: str | None = None
    max_epochs: int | None = None
    max_steps: int | None = None
    steps_per_epoch: int | None = None
    batch_size: int | None = None
    accumulation_steps: int | None = None
    seed: int | None = None
    checkpoint_dir: str | None = None
    eval_every: int | None = None
    run_dir: str | None = None
    world_size: int | None = None
    rank: int | None = None
    global_step: int = 0
    elapsed: float = 0.0


@dataclass(frozen=True)
class TrainBeginEvent:
    info: RunInfo
    total_steps: int | None = None


@dataclass(frozen=True)
class EpochBeginEvent:
    epoch: int
    total_epochs: int


@dataclass(frozen=True)
class StepEvent:
    global_step: int
    epoch: int
    epoch_step: int
    steps_per_epoch: int | None
    metrics: dict[str, float] = field(default_factory=dict)
    elapsed: float = 0.0


@dataclass(frozen=True)
class ValidationBeginEvent:
    epoch: int


@dataclass(frozen=True)
class ValidationEvent:
    epoch: int
    metrics: dict[str, float] = field(default_factory=dict)
    duration: float = 0.0


@dataclass(frozen=True)
class CheckpointEvent:
    path: str
    epoch: int
    global_step: int
    is_best: bool = False


@dataclass(frozen=True)
class EpochEndEvent:
    epoch: int
    train_metrics: dict = field(default_factory=dict)
    val_metrics: dict = field(default_factory=dict)
    best_metric: float | None = None
    duration: float = 0.0
    checkpoint_path: str | None = None


@dataclass(frozen=True)
class TrainEndEvent:
    total_epochs: int
    global_step: int
    duration: float = 0.0
    best_metric: float | None = None
    best_epoch: int | None = None
    history: list[dict] = field(default_factory=list)


def count_params(model: Any) -> tuple[int, int, int]:
    """Return (total, trainable, frozen) unique parameters; (0, 0, 0) for non-modules.

    Uninitialized lazy parameters (``nn.Lazy*`` before the first forward) have
    no size yet and are not counted.
    """
    from torch.nn.parameter import is_lazy

    if not callable(getattr(model, "parameters", None)):
        return (0, 0, 0)
    seen: set[int] = set()
    total = trainable = 0
    for p in model.parameters():
        if id(p) in seen or is_lazy(p):
            continue
        seen.add(id(p))
        total += p.numel()
        trainable += p.numel() if p.requires_grad else 0
    return (total, trainable, total - trainable)


def format_params(n: int | None) -> str:
    """Format param count like 12.4M. Never raises; returns '?' when unknown."""
    try:
        if n is None:
            return "?"
        n = int(n)
        if n < 0:
            return "?"
        if n >= 1_000_000_000:
            return f"{n / 1_000_000_000:.1f}B"
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n / 1_000:.1f}K"
        return str(n)
    except Exception:  # noqa: BLE001
        return "?"


def infer_batch_size(batch: Any) -> int | None:
    """Infer batch size from a batch object. Fails safely to None.

    Never consumes data: inspects shapes only.
    """
    try:
        if batch is None:
            return None
        # Direct tensor.
        try:
            import torch as _torch

            if isinstance(batch, _torch.Tensor):
                if batch.dim() == 0:
                    return None
                return int(batch.shape[0])
        except Exception:  # noqa: BLE001, S110 - batch size display is best-effort
            pass
        if isinstance(batch, dict):
            for v in batch.values():
                try:
                    import torch as _torch2

                    if isinstance(v, _torch2.Tensor):
                        if v.dim() == 0:
                            continue
                        return int(v.shape[0])
                except Exception:  # noqa: BLE001, S112 - batch size display is best-effort
                    continue
                # List-like fallback with len.
                try:
                    if hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
                        return len(v)  # type: ignore[arg-type]
                except Exception:  # noqa: BLE001, S112 - batch size display is best-effort
                    continue
            return None
        if isinstance(batch, (list, tuple)) and len(batch) > 0:
            return infer_batch_size(batch[0])
        # Generic sized object.
        try:
            if hasattr(batch, "__len__") and not isinstance(batch, (str, bytes)):
                return len(batch)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001, S110 - batch size display is best-effort
            pass
        return None
    except Exception:  # noqa: BLE001
        return None


def safe_loader_length(loader: Any) -> int | None:
    """Return len(loader) or None when unknown. Never raises."""
    if loader is None:
        return None
    try:
        return len(loader)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        return None


def safe_batch_size_from_loader(loader: Any) -> int | None:
    """Prefer loader.batch_size; never consumes a batch. Fails safely."""
    if loader is None:
        return None
    try:
        bs = getattr(loader, "batch_size", None)
        if bs is not None:
            return int(bs)
    except Exception:  # noqa: BLE001, S110 - batch size display is best-effort
        pass
    return None
