import math

import torch
from torch import nn


class SinkhornKnopp(nn.Module):
    component_id = "sinkhorn-knopp"

    def __init__(self, regularization, iterations=20):
        super().__init__()
        if type(regularization) not in (int, float):
            raise ValueError("regularization must be a finite positive scalar")
        if not math.isfinite(regularization) or regularization <= 0:
            raise ValueError("regularization must be a finite positive scalar")
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        self.regularization = float(regularization)
        self.iterations = iterations

    def forward(self, cost, source, target):
        if any(not torch.is_tensor(value) for value in (cost, source, target)):
            raise ValueError("cost, source, and target must be tensors")
        if cost.ndim not in (2, 3) or cost.shape[-2] != cost.shape[-1]:
            raise ValueError("cost must be a square matrix or a batch of square matrices")
        if source.ndim != cost.ndim - 1 or target.ndim != cost.ndim - 1:
            raise ValueError("source and target shapes must match the cost dimensions")
        if source.shape[-1] != cost.shape[-2] or target.shape[-1] != cost.shape[-1]:
            raise ValueError("source and target shapes must match the cost dimensions")
        if cost.ndim == 3 and (source.shape[0] != cost.shape[0] or target.shape[0] != cost.shape[0]):
            raise ValueError("batched source and target shapes must match the cost batch")
        if not cost.is_floating_point() or not source.is_floating_point() or not target.is_floating_point():
            raise ValueError("cost, source, and target must have floating point dtypes")
        if cost.device != source.device or cost.device != target.device:
            raise ValueError("cost, source, and target must be on the same device")
        if not torch.isfinite(cost).all().item() or not torch.isfinite(source).all().item() or not torch.isfinite(target).all().item():
            raise ValueError("cost and marginals must contain only finite values")
        if (source < 0).any().item() or (target < 0).any().item():
            raise ValueError("source and target marginals must be nonnegative")

        source = source.to(dtype=cost.dtype)
        target = target.to(dtype=cost.dtype)
        tolerance = 1e-5
        source_sums = source.sum(dim=-1)
        target_sums = target.sum(dim=-1)
        if not torch.isclose(source_sums, torch.ones_like(source_sums), atol=tolerance, rtol=tolerance).all().item():
            raise ValueError("source marginal must sum to one")
        if not torch.isclose(target_sums, torch.ones_like(target_sums), atol=tolerance, rtol=tolerance).all().item():
            raise ValueError("target marginal must sum to one")

        log_kernel = -self.regularization * cost
        if not torch.isfinite(log_kernel).all().item():
            raise ValueError("regularized costs must be finite")
        negative_infinity = cost.new_full((), -float("inf"))
        tiny = torch.finfo(cost.dtype).tiny
        log_source = torch.where(source > 0, source.clamp_min(tiny).log(), negative_infinity)
        log_target = torch.where(target > 0, target.clamp_min(tiny).log(), negative_infinity)
        log_v = torch.where(target > 0, torch.zeros_like(target), negative_infinity)

        for _ in range(self.iterations):
            log_u = log_source - torch.logsumexp(log_kernel + log_v.unsqueeze(-2), dim=-1)
            log_v = log_target - torch.logsumexp(log_kernel + log_u.unsqueeze(-1), dim=-2)

        log_u = log_source - torch.logsumexp(log_kernel + log_v.unsqueeze(-2), dim=-1)
        return torch.exp(log_u.unsqueeze(-1) + log_kernel + log_v.unsqueeze(-2))
