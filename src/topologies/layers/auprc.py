import torch
from torch import nn


class AveragePrecision(nn.Module):
    component_id = "auprc"
    def __init__(self):
        super().__init__()

    def forward(self, scores, labels):
        if scores.ndim != 1 or labels.ndim != 1:
            raise ValueError("scores and labels must be one-dimensional")
        if scores.shape != labels.shape:
            raise ValueError("scores and labels must have the same shape")
        if scores.numel() > 1 and torch.unique(scores).numel() != scores.numel():
            raise ValueError("scores must not contain ties")
        if not torch.all((labels == 0) | (labels == 1)):
            raise ValueError("labels must contain only 0 or 1")
        positive_count = labels.sum()
        if positive_count.item() == 0:
            raise ValueError("at least one positive label is required")
        ranked_labels = labels[torch.argsort(scores, descending=True)].to(dtype=scores.dtype)
        cumulative_positives = ranked_labels.cumsum(dim=0)
        ranks = torch.arange(1, labels.numel() + 1, device=scores.device, dtype=scores.dtype)
        precision_at_rank = cumulative_positives / ranks
        return (precision_at_rank * ranked_labels).sum() / positive_count.to(dtype=scores.dtype)
