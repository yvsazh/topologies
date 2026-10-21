"""Layer-wise learning-rate decay utility."""

from __future__ import annotations


def llrd(base_lr: float, depth: int, decay: float = 0.5) -> float:
    """Return the learning rate at a layer depth using exponential decay."""
    return base_lr * decay ** depth
