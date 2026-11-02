import math
import numbers

import torch
from torch import nn


class DBNet(nn.Module):
    component_id = "dbnet"

    def __init__(self, k=50.0):
        super().__init__()
        try:
            valid_k = (
                isinstance(k, numbers.Real)
                and not isinstance(k, bool)
                and math.isfinite(float(k))
                and k > 0
            )
        except (OverflowError, TypeError, ValueError):
            valid_k = False
        if not valid_k:
            raise ValueError("k must be a finite positive scalar")
        self.k = float(k)

    def forward(self, probability, threshold):
        if not isinstance(probability, torch.Tensor) or not isinstance(
            threshold, torch.Tensor
        ):
            raise TypeError("probability and threshold must be torch tensors")
        if probability.shape != threshold.shape:
            raise ValueError("probability and threshold must have the same shape")
        if not probability.is_floating_point() or not threshold.is_floating_point():
            raise ValueError("probability and threshold must be floating-point tensors")
        if probability.device != threshold.device:
            raise ValueError("probability and threshold must be on the same device")
        return torch.sigmoid(self.k * (probability - threshold))
