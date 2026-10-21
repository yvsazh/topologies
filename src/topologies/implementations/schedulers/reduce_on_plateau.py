"""Reduce learning rate when a monitored metric stops improving."""

import torch


def reduce_on_plateau(optimizer, total_steps, **kwargs):
    """Create a ReduceLROnPlateau scheduler with a no-argument ``step()``.

    PyTorch's scheduler needs a metric. The default metric is constant, making
    the scheduler useful in simple step-only training loops; callers may pass
    ``metric=...`` to ``step`` when they have a validation metric.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    kwargs.setdefault("patience", 0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)

    class _ReduceOnPlateau:
        def step(self, metric=0.0):
            scheduler.step(metric)

    return _ReduceOnPlateau()
