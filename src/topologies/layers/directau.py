import math
from numbers import Real

import torch
from torch import nn


class DirectAU(nn.Module):
    component_id = "directau"

    def __init__(self, gamma=1.0, reduction="mean"):
        super().__init__()
        if isinstance(gamma, bool) or not isinstance(gamma, Real):
            DirectAU._invalid("gamma must be a finite non-negative real number")
        try:
            gamma_value = float(gamma)
        except (OverflowError, ValueError):
            DirectAU._invalid("gamma must be a finite non-negative real number")
        if not math.isfinite(gamma_value) or gamma_value < 0:
            raise ValueError("gamma must be a finite non-negative real number")
        if not isinstance(reduction, str) or reduction not in ("mean", "none"):
            raise ValueError("reduction must be 'mean' or 'none'")
        self.gamma = gamma_value
        self.reduction = reduction

    @staticmethod
    def _invalid(message):
        raise ValueError(message)

    @staticmethod
    def _validate_pair(user_embeddings, item_embeddings):
        if not isinstance(user_embeddings, torch.Tensor) or not isinstance(item_embeddings, torch.Tensor):
            DirectAU._invalid("user and item embeddings must be tensors")
        if user_embeddings.ndim != 2 or item_embeddings.ndim != 2:
            raise ValueError("user and item embeddings must have rank-two [N, D] shapes")
        if user_embeddings.shape != item_embeddings.shape:
            raise ValueError("user and item embeddings must have matching [N, D] shapes")
        if user_embeddings.shape[0] < 1:
            raise ValueError("at least one positive pair is required")
        if user_embeddings.shape[1] < 1:
            raise ValueError("embedding dimension must be positive")
        if not user_embeddings.is_floating_point() or not item_embeddings.is_floating_point():
            raise ValueError("user and item embeddings must use floating-point dtypes")
        if user_embeddings.device != item_embeddings.device or user_embeddings.dtype != item_embeddings.dtype:
            raise ValueError("user and item embeddings must share a device and dtype")

    @staticmethod
    def _normalize(embeddings):
        return torch.nn.functional.normalize(embeddings, p=2, dim=-1)

    @staticmethod
    def alignment(user_embeddings, item_embeddings):
        DirectAU._validate_pair(user_embeddings, item_embeddings)
        users = DirectAU._normalize(user_embeddings)
        items = DirectAU._normalize(item_embeddings)
        return (users - items).square().sum(dim=-1).mean()

    @staticmethod
    def uniformity(embeddings):
        if not isinstance(embeddings, torch.Tensor):
            DirectAU._invalid("embeddings must be a tensor")
        if embeddings.ndim != 2:
            raise ValueError("embeddings must have a rank-two [N, D] shape")
        if embeddings.shape[0] < 2:
            raise ValueError("uniformity requires at least two embeddings")
        if embeddings.shape[1] < 1:
            raise ValueError("embedding dimension must be positive")
        if not embeddings.is_floating_point():
            raise ValueError("embeddings must use a floating-point dtype")
        normalized = DirectAU._normalize(embeddings)
        distances = torch.pdist(normalized, p=2).square()
        return torch.log(torch.exp(-2.0 * distances).mean())

    def forward(self, user_embeddings, item_embeddings):
        DirectAU._validate_pair(user_embeddings, item_embeddings)
        if user_embeddings.shape[0] < 2 and self.reduction == "mean":
            raise ValueError("at least two positive pairs are required for uniformity")
        users = DirectAU._normalize(user_embeddings)
        items = DirectAU._normalize(item_embeddings)
        alignment = (users - items).square().sum(dim=-1)
        if user_embeddings.shape[0] < 2:
            return alignment
        uniformity = 0.5 * (
            self.uniformity(user_embeddings) + self.uniformity(item_embeddings)
        )
        losses = alignment + self.gamma * uniformity
        return losses.mean() if self.reduction == "mean" else losses
