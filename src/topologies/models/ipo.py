import torch
from torch import nn


def ipo_loss(chosen_logp, rejected_logp, chosen_ref_logp, rejected_ref_logp, beta=0.1, reduction="mean"):
    if beta <= 0:
        raise ValueError("beta must be positive")
    gap = (chosen_logp - chosen_ref_logp) - (rejected_logp - rejected_ref_logp)
    target_margin = 1.0 / (2.0 * beta)
    losses = (gap - target_margin).square()
    if reduction == "mean":
        return losses.mean()
    if reduction == "sum":
        return losses.sum()
    if reduction == "none":
        return losses
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


class IPO(nn.Module):
    def __init__(self, beta=0.1, reduction="mean"):
        super().__init__()
        if beta <= 0:
            raise ValueError("beta must be positive")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("invalid reduction")
        self.beta = float(beta)
        self.reduction = reduction

    def forward(self, chosen_logp, rejected_logp, chosen_ref_logp, rejected_ref_logp):
        return ipo_loss(chosen_logp, rejected_logp, chosen_ref_logp, rejected_ref_logp, self.beta, self.reduction)
