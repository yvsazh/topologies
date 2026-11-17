import torch
from torch import nn


class STMoE(nn.Module):
    component_id = "st-moe"

    def __init__(self):
        super().__init__()

    def forward(self, x, reduction="mean"):
        if not isinstance(x, torch.Tensor):
            raise TypeError("router_logits must be a torch.Tensor")
        if x.ndim < 2:
            raise ValueError("router_logits must have token and expert dimensions")
        if any(size == 0 for size in x.shape):
            raise ValueError("router_logits dimensions must be non-empty")
        if reduction not in ("mean", "none"):
            raise ValueError("reduction must be 'mean' or 'none'")
        token_losses = torch.logsumexp(x, dim=-1).square()
        return token_losses.mean() if reduction == "mean" else token_losses
