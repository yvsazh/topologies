import torch
from torch import nn


class OptNet(nn.Module):
    component_id = "optnet"

    def __init__(self, *args, **kwargs):
        if args or kwargs:
            raise ValueError("OptNet does not accept constructor arguments")
        super().__init__()

    def forward(self, x, a):
        if not isinstance(x, torch.Tensor) or not isinstance(a, torch.Tensor):
            raise TypeError("x and a must be tensors")
        if x.ndim < 1 or a.ndim < 1:
            raise ValueError("x and a must have at least one dimension")
        if x.shape != a.shape:
            raise ValueError("x and a must have the same shape")
        if x.shape[-1] < 1:
            raise ValueError("the final dimension must be non-empty")
        if not torch.is_floating_point(x) or not torch.is_floating_point(a):
            raise ValueError("x and a must have floating-point dtypes")
        if x.dtype != a.dtype:
            raise ValueError("x and a must have the same dtype")
        if x.device != a.device:
            raise ValueError("x and a must be on the same device")
        if not torch.isfinite(x).all().item() or not torch.isfinite(a).all().item():
            raise ValueError("x and a must contain only finite values")
        if not torch.all(torch.any(a != 0, dim=-1)).item():
            raise ValueError("each constraint vector a must be nonzero")
        denominator = (a * a).sum(dim=-1, keepdim=True)
        if not torch.isfinite(denominator).all().item() or torch.any(denominator == 0).item():
            raise ValueError("each constraint vector must have a finite positive squared norm")
        correction = (1 - (a * x).sum(dim=-1, keepdim=True)) / denominator
        result = x + correction * a
        if not torch.isfinite(result).all().item():
            raise ValueError("the projection result must be finite")
        return result
