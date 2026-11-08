import torch
from torch import nn


class MCDDiscrepancy(nn.Module):
    component_id = "mcd"

    def __init__(self, reduction="mean"):
        super().__init__()
        if not isinstance(reduction, str) or reduction not in ("mean", "none"):
            raise ValueError("reduction must be 'mean' or 'none'")
        self.reduction = reduction

    def forward(self, logits1, logits2):
        if not isinstance(logits1, torch.Tensor) or not isinstance(logits2, torch.Tensor):
            raise TypeError("logits1 and logits2 must be tensors")
        if logits1.ndim != 2 or logits2.ndim != 2:
            raise ValueError("logits1 and logits2 must be 2D tensors shaped [B, K]")
        if logits1.shape != logits2.shape:
            raise ValueError("logits1 and logits2 must have the same shape")
        if logits1.shape[0] < 1:
            raise ValueError("the batch dimension must be positive")
        if logits1.shape[1] < 2:
            raise ValueError("logits must contain at least two classes")
        if not logits1.is_floating_point() or not logits2.is_floating_point():
            raise ValueError("logits1 and logits2 must have floating-point values")
        if logits1.device != logits2.device:
            raise ValueError("logits1 and logits2 must be on the same device")

        probabilities1 = torch.softmax(logits1, dim=1)
        probabilities2 = torch.softmax(logits2, dim=1)
        discrepancies = (probabilities1 - probabilities2).abs().mean(dim=1)
        return discrepancies.mean() if self.reduction == "mean" else discrepancies
