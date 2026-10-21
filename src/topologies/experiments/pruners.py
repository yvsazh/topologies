"""Early stopping across trials: the ``Pruner`` protocol, ``MedianPruner`` and the
successive-halving rung schedule.

    Experiment(..., prune="median")                       # MedianPruner(1, 4)
    Experiment(..., prune=MedianPruner(warmup_epochs=2))
    Experiment(..., strategy="halving", eta=3, train_for="9 epochs")

A pruner decides at every epoch end:
``should_prune(trial_id, epoch, value, history_by_trial) -> bool`` where
``history_by_trial`` maps other trial ids to ``{epoch: goal value}``.
"""

from __future__ import annotations

import math
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Pruner(Protocol):
    def should_prune(self, trial_id: str, epoch: int, value: float,
                     history_by_trial: dict[str, dict[int, float]]) -> bool: ...


class MedianPruner:
    """Stop a trial whose goal is worse than the median of other trials at the same epoch.

    Not before ``warmup_epochs`` epochs, and only when at least ``min_trials``
    other trials reported that epoch. ``mode`` (``"min"``/``"max"``) is set
    from the experiment goal when left as ``None``.
    """

    def __init__(self, warmup_epochs: int = 1, min_trials: int = 4,
                 mode: str | None = None) -> None:
        self.warmup_epochs = int(warmup_epochs)
        self.min_trials = int(min_trials)
        self.mode = mode

    def should_prune(self, trial_id: str, epoch: int, value: float,
                     history_by_trial: dict[str, dict[int, float]]) -> bool:
        if epoch < self.warmup_epochs or value is None or not math.isfinite(value):
            return False
        others = []
        for tid, hist in history_by_trial.items():
            if tid == trial_id:
                continue
            v = hist.get(epoch, hist.get(str(epoch))) if isinstance(hist, dict) else None
            if v is not None and math.isfinite(float(v)):
                others.append(float(v))
        if len(others) < self.min_trials:
            return False
        others.sort()
        n = len(others)
        med = others[n // 2] if n % 2 else (others[n // 2 - 1] + others[n // 2]) / 2
        return value > med if (self.mode or "min") == "min" else value < med

    def __repr__(self) -> str:
        return f"MedianPruner(warmup_epochs={self.warmup_epochs}, min_trials={self.min_trials})"


def halving_rungs(max_epochs: int, *, eta: int = 3, min_epochs: int = 1) -> list[int]:
    """Epoch budgets per rung: ``min, min*eta, ...`` capped by (and ending at) ``max``."""
    if eta < 2:
        raise ValueError("eta must be >= 2")
    max_epochs, b = int(max_epochs), max(1, int(min_epochs))
    rungs = []
    while b < max_epochs:
        rungs.append(b)
        b *= eta
    rungs.append(max_epochs)
    return rungs


def keep_count(n: int, eta: int) -> int:
    """Configs promoted from a rung of ``n``: ``ceil(n / eta)``."""
    return max(1, math.ceil(n / eta))


def rank_configs(scores: dict[Any, float | None], mode: str) -> list[Any]:
    """Config keys best first (configs without a score last)."""
    sign = 1.0 if mode == "min" else -1.0
    return sorted(scores, key=lambda k: (scores[k] is None or not math.isfinite(scores[k]),
                                         sign * (scores[k] or 0.0)))


__all__ = ["MedianPruner", "Pruner", "halving_rungs", "keep_count", "rank_configs"]
