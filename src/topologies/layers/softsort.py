from math import isfinite

import torch
from torch import nn


class SoftSort(nn.Module):
    component_id = "softsort"

    def __init__(self, tau):
        super().__init__()
        if type(tau) not in (int, float):
            raise ValueError("tau must be a finite positive number")
        try:
            tau = float(tau)
        except (OverflowError, ValueError):
            raise ValueError("tau must be a finite positive number") from None
        if not isfinite(tau) or tau <= 0.0:
            raise ValueError("tau must be a finite positive number")
        self.tau = tau

    def forward(self, scores):
        if not torch.is_tensor(scores):
            raise ValueError("scores must be a floating-point tensor with shape (B, N, 1)")
        if scores.ndim != 3 or scores.shape[-1] != 1:
            raise ValueError("scores must have shape (B, N, 1)")
        if scores.shape[0] < 1 or scores.shape[1] < 1:
            raise ValueError("scores must have positive batch and sequence dimensions")
        if not scores.is_floating_point():
            raise ValueError("scores must have a floating-point dtype")

        sorted_scores = scores.sort(dim=1, descending=True).values
        logits = -(scores.transpose(1, 2) - sorted_scores).abs() / self.tau
        return torch.softmax(logits, dim=-1)
