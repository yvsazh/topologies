from math import isfinite

import torch
from torch import nn


class NeuralSort(nn.Module):
    component_id = "neural-sort"

    def __init__(self, tau=1.0):
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

        values = scores.squeeze(-1)
        sequence_length = values.shape[1]
        pairwise_distances = torch.abs(values.unsqueeze(2) - values.unsqueeze(1))
        distance_sums = pairwise_distances.sum(dim=-1)
        ranks = torch.arange(sequence_length, device=scores.device, dtype=scores.dtype)
        rank_coefficients = (sequence_length - 1 - 2 * ranks).view(1, sequence_length, 1)
        logits = rank_coefficients * values.unsqueeze(1) - distance_sums.unsqueeze(1)
        return torch.softmax(logits / self.tau, dim=-1)
