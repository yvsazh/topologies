import math
import numbers

import torch
import torch.nn.functional as F


def varifocal_loss(
    logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params
):
    if ignore_index is not None:
        raise ValueError("ignore_index is unsupported for continuous quality targets")
    if not isinstance(reduction, str) or reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")

    unexpected = set(params) - {"alpha", "gamma", "avg_factor"}
    if unexpected:
        raise ValueError(f"unexpected parameters: {', '.join(sorted(unexpected))}")

    alpha = params.get("alpha", 0.75)
    valid_alpha = isinstance(alpha, numbers.Real) and not isinstance(alpha, bool)
    if valid_alpha:
        try:
            valid_alpha = math.isfinite(alpha) and alpha >= 0
        except (OverflowError, TypeError, ValueError):
            valid_alpha = False
    if not valid_alpha:
        raise ValueError("alpha must be a finite nonnegative real number")
    alpha = float(alpha)

    gamma = params.get("gamma", 2.0)
    valid_gamma = isinstance(gamma, numbers.Real) and not isinstance(gamma, bool)
    if valid_gamma:
        try:
            valid_gamma = math.isfinite(gamma) and gamma >= 0
        except (OverflowError, TypeError, ValueError):
            valid_gamma = False
    if not valid_gamma:
        raise ValueError("gamma must be a finite nonnegative real number")
    gamma = float(gamma)

    avg_factor = params.get("avg_factor")
    if avg_factor is not None:
        if reduction != "mean":
            raise ValueError("avg_factor is only supported with reduction='mean'")
        valid_avg_factor = isinstance(avg_factor, numbers.Real) and not isinstance(
            avg_factor, bool
        )
        if valid_avg_factor:
            try:
                valid_avg_factor = math.isfinite(avg_factor) and avg_factor > 0
            except (OverflowError, TypeError, ValueError):
                valid_avg_factor = False
        if not valid_avg_factor:
            raise ValueError("avg_factor must be a finite positive real number")
        avg_factor = float(avg_factor)

    if (
        not isinstance(logits_or_prediction, torch.Tensor)
        or not isinstance(targets, torch.Tensor)
        or not logits_or_prediction.is_floating_point()
        or not targets.is_floating_point()
    ):
        raise ValueError("predictions and targets must be floating-point tensors")
    if logits_or_prediction.shape != targets.shape:
        raise ValueError("predictions and targets must have identical shapes")
    if logits_or_prediction.numel() == 0:
        raise ValueError("predictions and targets must be nonempty")
    if logits_or_prediction.device != targets.device:
        raise ValueError("predictions and targets must be on the same device")
    if not torch.isfinite(logits_or_prediction).all():
        raise ValueError("predictions must contain only finite values")
    if ((logits_or_prediction < 0) | (logits_or_prediction > 1)).any():
        raise ValueError("predictions must contain values in [0, 1]")
    if not torch.isfinite(targets).all():
        raise ValueError("targets must contain only finite values")
    if ((targets < 0) | (targets > 1)).any():
        raise ValueError("targets must contain values in [0, 1]")

    targets = targets.to(dtype=logits_or_prediction.dtype)
    positive = targets > 0
    negative_weight = alpha * logits_or_prediction.pow(gamma)
    weights = torch.where(positive, targets, negative_weight)
    elementwise = weights * F.binary_cross_entropy(
        logits_or_prediction, targets, reduction="none"
    )

    if reduction == "none":
        return elementwise
    if reduction == "sum":
        return elementwise.sum()
    if avg_factor is not None:
        return elementwise.sum() / avg_factor
    return elementwise.mean()
