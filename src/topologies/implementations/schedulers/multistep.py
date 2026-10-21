"""Multi-step learning-rate scheduler factory."""

import torch


def multistep(optimizer, total_steps, **kwargs):
    """Create a MultiStepLR scheduler.

    By default the learning rate is reduced once halfway through the run;
    ``milestones`` and ``gamma`` can be supplied as for PyTorch's MultiStepLR.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    milestones = kwargs.pop("milestones", [max(1, total_steps // 2)])
    gamma = kwargs.pop("gamma", 0.1)
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported MultiStep scheduler options: {unsupported}")
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=gamma
    )
