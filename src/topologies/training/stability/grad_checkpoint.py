"""Layer selection for gradient checkpointing."""

from __future__ import annotations


def grad_checkpoint(n_layers: int, every: int = 2) -> list[int]:
    """Return zero-based layer indexes to checkpoint at the given interval.

    Checkpointing is applied to every ``every``-th layer, beginning with
    index ``every - 1``.
    """
    if n_layers < 0:
        raise ValueError("n_layers must be non-negative")
    if every <= 0:
        raise ValueError("every must be positive")
    return list(range(every - 1, n_layers, every))
