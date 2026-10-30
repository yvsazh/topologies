import torch
from torch import nn


class UncorrectedSampledSetCrossEntropy(nn.Module):
    component_id = "sampled-softmax"

    def __init__(self, reduction="mean"):
        super().__init__()
        if not isinstance(reduction, str) or reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")
        self.reduction = reduction

    def forward(self, positive_logits, negative_logits):
        positive_logits = torch.as_tensor(positive_logits)
        negative_logits = torch.as_tensor(
            negative_logits,
            device=positive_logits.device,
            dtype=positive_logits.dtype,
        )
        if positive_logits.ndim < 1:
            raise ValueError("positive_logits must have at least one dimension")
        if not positive_logits.is_floating_point():
            raise ValueError("positive_logits must have a floating-point dtype")
        if negative_logits.ndim != positive_logits.ndim + 1:
            raise ValueError("negative_logits must add a final negative-sample dimension")
        if negative_logits.shape[:-1] != positive_logits.shape or negative_logits.shape[-1] < 1:
            raise ValueError("negative_logits must match positive_logits and contain at least one negative")
        logits = torch.cat((positive_logits.unsqueeze(-1), negative_logits), dim=-1)
        losses = torch.logsumexp(logits, dim=-1) - positive_logits
        if self.reduction == "mean":
            return losses.mean()
        if self.reduction == "sum":
            return losses.sum()
        return losses
