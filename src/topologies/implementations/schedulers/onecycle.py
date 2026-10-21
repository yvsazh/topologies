"""One-cycle learning-rate scheduler factory."""

import torch


def onecycle(optimizer, total_steps, **kwargs):
    """Create PyTorch's one-cycle learning-rate scheduler for ``optimizer``."""
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer, total_steps=total_steps, **kwargs
    )
