"""Cosine-annealing learning-rate schedule with warm restarts."""

import torch


def cosine_restarts(optimizer, total_steps, **kwargs):
    """Create a cosine-annealing scheduler with warm restarts.

    ``total_steps`` is the first restart period by default. Pass ``t_0`` or
    ``t_mult`` to configure restart periods explicitly.
    """
    t_0 = kwargs.pop("t_0", total_steps)
    t_mult = kwargs.pop("t_mult", 1)
    eta_min = kwargs.pop("eta_min", 0.0)
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise TypeError(f"Cosine restarts does not support options: {unsupported}")
    if t_0 <= 0:
        raise ValueError("total_steps/t_0 must be positive")
    if t_mult < 1:
        raise ValueError("t_mult must be at least 1")
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=t_0, T_mult=t_mult, eta_min=eta_min
    )
