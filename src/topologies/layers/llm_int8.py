import math
import numbers

import torch
from torch import nn


def _valid_threshold(value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return False
    try:
        value = float(value)
    except (OverflowError, ValueError):
        return False
    return math.isfinite(value) and value >= 0


class LLMInt8Linear(nn.Module):
    component_id = "llm-int8"

    def __init__(self, in_features, out_features, threshold=6.0):
        super().__init__()
        if type(in_features) is not int or in_features <= 0:
            raise ValueError("in_features must be a positive integer")
        if type(out_features) is not int or out_features <= 0:
            raise ValueError("out_features must be a positive integer")
        if not _valid_threshold(threshold):
            raise ValueError("threshold must be a finite nonnegative real number")
        threshold = float(threshold)
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("input must be a floating-point tensor")
        if x.ndim == 0:
            raise ValueError("input must have a final feature dimension")
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"input final dimension must equal in_features ({self.in_features})"
            )
        if not x.is_floating_point():
            raise ValueError("input must be a floating-point tensor")
        if not torch.isfinite(x).all():
            raise ValueError("input must contain only finite values")
        if x.device != self.weight.device:
            raise ValueError("input and weight must be on the same device")

        flat = x.reshape(-1, self.in_features)
        weight = self.weight.to(dtype=x.dtype)
        if flat.shape[0]:
            outlier_mask = flat.abs().amax(dim=0) >= self.threshold
        else:
            outlier_mask = torch.zeros(
                self.in_features, dtype=torch.bool, device=x.device
            )
        regular_mask = ~outlier_mask
        output = x.new_zeros((flat.shape[0], self.out_features))

        if outlier_mask.any():
            output = output + flat[:, outlier_mask] @ weight[:, outlier_mask].transpose(0, 1)

        if regular_mask.any():
            regular_x = flat[:, regular_mask]
            regular_weight = weight[:, regular_mask]
            input_scale = regular_x.abs().amax(dim=1, keepdim=True)
            weight_scale = regular_weight.abs().amax(dim=1, keepdim=True)
            safe_input_scale = torch.where(input_scale > 0, input_scale, 1.0)
            safe_weight_scale = torch.where(weight_scale > 0, weight_scale, 1.0)
            quantized_x = torch.round(127.0 * regular_x / safe_input_scale)
            quantized_weight = torch.round(
                127.0 * regular_weight / safe_weight_scale
            )
            quantized_x = quantized_x.clamp(-127, 127).to(torch.int8)
            quantized_weight = quantized_weight.clamp(-127, 127).to(torch.int8)
            integer_product = quantized_x.to(torch.int32) @ quantized_weight.to(
                torch.int32
            ).transpose(0, 1)
            scale = input_scale * weight_scale.transpose(0, 1) / (127.0**2)
            output = output + integer_product.to(x.dtype) * scale

        return output.reshape(*x.shape[:-1], self.out_features)
