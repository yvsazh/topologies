import math
import numbers
import sys

import torch


def score_matching_loss(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    sigma = params.pop("sigma", 0.01)
    if params:
        raise ValueError(f"unexpected parameters: {', '.join(sorted(params))}")
    if ignore_index is not None:
        raise ValueError("ignore_index must be None for score matching")
    if not isinstance(reduction, str) or reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    valid_sigma_type = not isinstance(sigma, bool) and isinstance(sigma, numbers.Real)
    if not valid_sigma_type:
        raise ValueError("sigma must be a finite positive real number")
    if sigma <= 0 or sigma > sys.float_info.max:
        raise ValueError("sigma must be a finite positive real number")
    sigma_value = float(sigma)
    if not math.isfinite(sigma_value) or sigma_value <= 0:
        raise ValueError("sigma must be a finite positive real number")
    if not isinstance(logits_or_prediction, torch.Tensor) or not logits_or_prediction.is_floating_point():
        raise ValueError("logits_or_prediction must be a floating-point tensor")
    if not isinstance(targets, torch.Tensor) or not targets.is_floating_point():
        raise ValueError("targets must be a floating-point tensor")
    if logits_or_prediction.ndim < 1 or any(size <= 0 for size in logits_or_prediction.shape):
        raise ValueError("logits_or_prediction must have a nonempty batch and event shape [B, ...]")
    if targets.shape != logits_or_prediction.shape:
        raise ValueError("logits_or_prediction and targets must have identical shapes")
    if targets.dtype != logits_or_prediction.dtype or targets.device != logits_or_prediction.device:
        raise ValueError("logits_or_prediction and targets must have matching dtype and device")

    batch_size = logits_or_prediction.shape[0]
    per_sample = 0.5 * (logits_or_prediction + targets / sigma_value).square().reshape(batch_size, -1).sum(dim=1)
    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    return per_sample.mean()
