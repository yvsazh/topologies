import math
import numbers

import torch


def wing_loss(logits_or_prediction, targets, *, ignore_index=None, reduction="sum", **params):
    if ignore_index is not None:
        raise ValueError("ignore_index must be None for Wing loss regression targets")
    if not isinstance(reduction, str) or reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")

    width = params.pop("width", 10.0)
    epsilon = params.pop("epsilon", 2.0)
    if params:
        raise ValueError(f"unexpected parameters: {', '.join(sorted(params))}")
    if (
        isinstance(width, bool)
        or not isinstance(width, numbers.Real)
        or not math.isfinite(float(width))
        or width < 0
    ):
        raise ValueError("width must be a finite nonnegative real number")
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, numbers.Real)
        or not math.isfinite(float(epsilon))
        or epsilon <= 0
    ):
        raise ValueError("epsilon must be a finite positive real number")
    if not isinstance(logits_or_prediction, torch.Tensor) or not logits_or_prediction.is_floating_point():
        raise ValueError("logits_or_prediction must be a floating-point tensor")
    if not isinstance(targets, torch.Tensor) or not targets.is_floating_point():
        raise ValueError("targets must be a floating-point tensor")
    if logits_or_prediction.shape != targets.shape:
        raise ValueError("logits_or_prediction and targets must have identical shapes")
    if logits_or_prediction.device != targets.device:
        raise ValueError("logits_or_prediction and targets must be on the same device")

    absolute_error = (logits_or_prediction - targets).abs()
    width = float(width)
    epsilon = float(epsilon)
    offset = width - width * math.log1p(width / epsilon)
    losses = torch.where(
        absolute_error < width,
        width * torch.log1p(absolute_error / epsilon),
        absolute_error - offset,
    )
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.mean()
