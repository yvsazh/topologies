"""Cosine annealing learning-rate schedule with a minimum learning rate."""

import torch


def cosine_min(optimizer, total_steps, **kwargs):
    """Create a cosine-annealing scheduler for ``total_steps``.

    ``eta_min`` sets the minimum learning rate and defaults to zero.
    """
    eta_min = kwargs.pop("eta_min", 0.0)
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise TypeError(f"Cosine-min does not support options: {unsupported}")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=eta_min
    )
