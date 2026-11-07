import torch
from torch import nn
from torch.nn import functional as F


class IBP(nn.Module):
    component_id = "ibp"

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        if type(in_features) is not int or in_features <= 0:
            raise ValueError("in_features must be a positive integer")
        if type(out_features) is not int or out_features <= 0:
            raise ValueError("out_features must be a positive integer")
        if type(bias) is not bool:
            raise ValueError("bias must be a bool")
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, lower, upper, output_format="nested"):
        if output_format not in ("nested", "stacked"):
            raise ValueError("output_format must be 'nested' or 'stacked'")
        if not isinstance(lower, torch.Tensor) or not isinstance(upper, torch.Tensor):
            raise TypeError("lower and upper bounds must be tensors")
        if lower.shape != upper.shape:
            raise ValueError("lower and upper bounds must have the same shape")
        if lower.ndim == 0 or lower.shape[-1] != self.in_features:
            raise ValueError(f"the final input dimension must equal in_features ({self.in_features})")
        if lower.device != upper.device:
            raise ValueError("lower and upper bounds must be on the same device")
        if lower.dtype != upper.dtype or not lower.is_floating_point():
            raise ValueError("lower and upper bounds must have the same floating dtype")
        if lower.device != self.linear.weight.device:
            raise ValueError("input bounds and layer parameters must be on the same device")
        if lower.dtype != self.linear.weight.dtype:
            raise ValueError("input bounds and layer parameters must have the same dtype")
        if not torch.isfinite(lower).all().item() or not torch.isfinite(upper).all().item():
            raise ValueError("lower and upper bounds must be finite")
        if torch.any(lower > upper).item():
            raise ValueError("each lower bound must be less than or equal to its upper bound")

        center = (lower + upper) / 2
        radius = (upper - lower) / 2
        center_out = F.linear(center, self.linear.weight, self.linear.bias)
        radius_out = F.linear(radius, torch.abs(self.linear.weight))
        lower_out = center_out - radius_out
        upper_out = center_out + radius_out
        if output_format == "stacked":
            return torch.stack((lower_out, upper_out), dim=-1)
        return lower_out, upper_out
