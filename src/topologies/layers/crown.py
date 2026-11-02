import torch
from torch import nn


class CROWN(nn.Module):
    component_id = "crown"

    def forward(self, lower, upper, output_format="nested"):
        if output_format not in ("nested", "stacked"):
            raise ValueError("output_format must be 'nested' or 'stacked'")
        if not isinstance(lower, torch.Tensor) or not isinstance(upper, torch.Tensor):
            raise TypeError("lower and upper bounds must be tensors")
        if lower.shape != upper.shape:
            raise ValueError("lower and upper bounds must have the same shape")
        if lower.device != upper.device:
            raise ValueError("lower and upper bounds must be on the same device")
        if lower.dtype != upper.dtype or not lower.is_floating_point():
            raise ValueError("lower and upper bounds must have the same floating dtype")
        if not torch.isfinite(lower).all().item() or not torch.isfinite(upper).all().item():
            raise ValueError("lower and upper bounds must be finite")
        if torch.any(lower > upper).item():
            raise ValueError("each lower bound must be less than or equal to its upper bound")

        positive = lower >= 0
        negative = upper <= 0
        crossing = ~(positive | negative)
        denominator = torch.where(crossing, upper - lower, torch.ones_like(lower))
        upper_slope = upper / denominator
        upper_intercept = -(lower * upper) / denominator

        lower_slope = torch.where(negative, torch.zeros_like(lower), torch.ones_like(lower))
        lower_intercept = torch.zeros_like(lower)
        upper_slope = torch.where(
            positive,
            torch.ones_like(upper),
            torch.where(negative, torch.zeros_like(upper), upper_slope),
        )
        upper_intercept = torch.where(crossing, upper_intercept, torch.zeros_like(upper))

        if output_format == "stacked":
            return torch.stack((lower_slope, lower_intercept, upper_slope, upper_intercept), dim=-1)
        return (lower_slope, lower_intercept), (upper_slope, upper_intercept)
