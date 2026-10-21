"""A lightweight ReLoRA-style optimizer factory.

ReLoRA periodically restarts low-rank updates after merging them into the
base weights. Weight merging is model-specific, so this optimizer provides
the Adam update step; callers perform any adapter merge/reinitialization.
"""

from torch.optim import Adam


class ReLoRA(Adam):
    """Adam optimizer used for the low-rank update phases of ReLoRA."""


def relora(params, *, lr, weight_decay=0.0, **backend_options):
    """Construct a ReLoRA optimizer, forwarding options to ``torch.optim.Adam``."""
    return ReLoRA(params, lr=lr, weight_decay=weight_decay, **backend_options)
