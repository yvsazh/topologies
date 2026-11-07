import torch
from torch import nn


class GraphMVP(nn.Module):
    component_id = "graphmvp"

    def __init__(self, reduction="mean"):
        super().__init__()
        if reduction not in ("mean", "none"):
            raise ValueError("reduction must be 'mean' or 'none'")
        self.reduction = reduction

    def forward(self, positive_2d, positive_3d, negative_2d, negative_3d):
        embeddings = (positive_2d, positive_3d, negative_2d, negative_3d)
        names = ("positive_2d", "positive_3d", "negative_2d", "negative_3d")
        for name, embedding in zip(names, embeddings):
            if not embedding.is_floating_point():
                raise ValueError(f"{name} must have a floating-point dtype")

        if positive_2d.ndim != 2 or positive_3d.ndim != 2:
            raise ValueError("positive embeddings must have shape [B,D]")
        if positive_2d.shape != positive_3d.shape:
            raise ValueError("positive embeddings must have matching [B,D] shapes")
        batch_size, feature_dim = positive_2d.shape
        if batch_size < 1 or feature_dim < 1:
            raise ValueError("positive embeddings must have non-empty batch and feature dimensions")

        if negative_2d.ndim != 3 or negative_3d.ndim != 3:
            raise ValueError("negative embeddings must have shape [B,K,D]")
        expected_shape = (batch_size, negative_2d.shape[1], feature_dim)
        if negative_2d.shape != expected_shape or negative_3d.shape != expected_shape:
            raise ValueError("negative embeddings must have matching [B,K,D] shapes aligned with positives")
        if negative_2d.shape[1] < 1:
            raise ValueError("each positive pair must have at least one negative")

        first = positive_2d
        for name, embedding in zip(names[1:], embeddings[1:]):
            if embedding.device != first.device:
                raise ValueError("all embeddings must be on the same device")
            if embedding.dtype != first.dtype:
                raise ValueError("all embeddings must have the same dtype")

        positive_scores = (positive_2d * positive_3d).sum(dim=-1)
        negative_2d_scores = torch.einsum("bkd,bd->bk", negative_2d, positive_3d)
        negative_3d_scores = torch.einsum("bd,bkd->bk", positive_2d, negative_3d)
        logits_2d = torch.cat((positive_scores.unsqueeze(1), negative_2d_scores), dim=1)
        logits_3d = torch.cat((positive_scores.unsqueeze(1), negative_3d_scores), dim=1)
        loss_2d = torch.logsumexp(logits_2d, dim=1) - positive_scores
        loss_3d = torch.logsumexp(logits_3d, dim=1) - positive_scores
        loss = (loss_2d + loss_3d) * 0.5
        return loss.mean() if self.reduction == "mean" else loss
