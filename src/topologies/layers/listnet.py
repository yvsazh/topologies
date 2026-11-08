import torch
from torch import nn


class ListNet(nn.Module):
    component_id = "listnet"

    def __init__(self, reduction="mean"):
        super().__init__()
        if not isinstance(reduction, str) or reduction not in ("none", "mean", "sum"):
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")
        self.reduction = reduction

    def forward(self, labels, scores):
        scores = torch.as_tensor(scores)
        labels = torch.as_tensor(labels, device=scores.device)
        if labels.shape != scores.shape:
            raise ValueError("labels and scores must have the same shape")
        if labels.ndim not in (1, 2) or any(size == 0 for size in labels.shape):
            raise ValueError("labels and scores must have shape [L] or [B, L] with nonempty dimensions")
        if labels.is_complex() or scores.is_complex():
            raise ValueError("labels and scores must be real-valued")
        if not labels.is_floating_point():
            labels = labels.to(torch.get_default_dtype())
        if not scores.is_floating_point():
            scores = scores.to(labels.dtype)
        dtype = torch.promote_types(labels.dtype, scores.dtype)
        labels = labels.to(dtype=dtype)
        scores = scores.to(dtype=dtype)
        target_probabilities = torch.softmax(labels, dim=-1)
        per_list_losses = -(target_probabilities * torch.log_softmax(scores, dim=-1)).sum(dim=-1)
        if self.reduction == "none":
            return per_list_losses
        if self.reduction == "sum":
            return per_list_losses.sum()
        return per_list_losses.mean()
