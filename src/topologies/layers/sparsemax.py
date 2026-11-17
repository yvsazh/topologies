import torch
from torch import nn


class Sparsemax(nn.Module):
    component_id = "sparsemax"

    def __init__(self, mode="sparsemax", dim=-1):
        super().__init__()
        if mode not in ("sparsemax", "entmax15"):
            raise ValueError("mode must be 'sparsemax' or 'entmax15'")
        if isinstance(dim, bool) or not isinstance(dim, int):
            error_type = ValueError
            raise error_type("dim must be an integer")
        self.mode = mode
        self.dim = dim

    def forward(self, logits):
        if not isinstance(logits, torch.Tensor):
            raise TypeError("logits must be a torch.Tensor")
        if logits.ndim == 0:
            raise ValueError("logits must have at least one dimension")
        if not torch.is_floating_point(logits):
            raise ValueError("logits must have a floating-point dtype")
        if self.dim < -logits.ndim or self.dim >= logits.ndim:
            raise ValueError("dim must select an axis within the logits rank")
        dim = self.dim % logits.ndim
        if logits.shape[dim] == 0:
            raise ValueError("the normalized dimension must be non-empty")
        output_dtype = logits.dtype
        work = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
        if self.mode == "sparsemax":
            output = self._sparsemax(work, dim)
        else:
            output = self._entmax15(work, dim)
        return output.to(dtype=output_dtype)

    def _sparsemax(self, logits, dim):
        shifted = logits - logits.amax(dim=dim, keepdim=True)
        sorted_logits = torch.sort(shifted, dim=dim, descending=True).values
        shape = [1] * logits.ndim
        shape[dim] = logits.shape[dim]
        ranks = torch.arange(1, logits.shape[dim] + 1, dtype=logits.dtype, device=logits.device).view(shape)
        cumulative = sorted_logits.cumsum(dim=dim)
        support = 1 + ranks * sorted_logits > cumulative
        support_size = support.sum(dim=dim, keepdim=True).clamp(min=1)
        threshold_sum = cumulative.gather(dim, support_size - 1)
        threshold = (threshold_sum - 1) / support_size.to(dtype=logits.dtype)
        return (shifted - threshold).clamp_min(0)

    def _entmax15(self, logits, dim):
        scaled = logits / 2
        scaled = scaled - scaled.amax(dim=dim, keepdim=True)
        sorted_logits = torch.sort(scaled, dim=dim, descending=True).values
        shape = [1] * logits.ndim
        shape[dim] = logits.shape[dim]
        ranks = torch.arange(1, logits.shape[dim] + 1, dtype=logits.dtype, device=logits.device).view(shape)
        cumulative = sorted_logits.cumsum(dim=dim)
        means = cumulative / ranks
        squared_deviations = sorted_logits.square().cumsum(dim=dim) - cumulative.square() / ranks
        delta = 1 - squared_deviations
        thresholds = means - (delta.clamp_min(0) / ranks).sqrt()
        support = (delta >= 0) & (thresholds <= sorted_logits)
        support_size = support.sum(dim=dim, keepdim=True).clamp(min=1)
        threshold = thresholds.gather(dim, support_size - 1)
        return (scaled - threshold).clamp_min(0).square()
