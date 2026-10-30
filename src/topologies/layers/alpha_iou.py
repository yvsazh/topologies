import math
from numbers import Real

import torch
from torch import nn


def _is_valid_alpha_type(value):
    return isinstance(value, Real) and not isinstance(value, bool)


class AlphaIou(nn.Module):
    component_id = "alpha-iou"

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        if not _is_valid_alpha_type(alpha):
            raise ValueError("alpha must be a finite scalar greater than or equal to zero")
        try:
            alpha = float(alpha)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError("alpha must be a finite scalar greater than or equal to zero") from error
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be a finite scalar greater than or equal to zero")
        self.alpha = alpha

    def forward(self, iou: torch.Tensor) -> torch.Tensor:
        if not isinstance(iou, torch.Tensor) or not iou.is_floating_point():
            raise ValueError("IoU must be a floating point tensor")
        if not torch.isfinite(iou).all().item():
            raise ValueError("IoU values must be finite")
        if ((iou < 0) | (iou > 1)).any().item():
            raise ValueError("IoU values must be in the range [0, 1]")
        if self.alpha == 0:
            return -torch.log(iou)
        return 1 - torch.pow(iou, self.alpha)
