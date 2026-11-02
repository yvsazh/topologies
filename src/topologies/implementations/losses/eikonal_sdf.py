import math
import numbers

import torch


def eikonal_sdf(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if ignore_index is not None:
        raise ValueError("ignore_index must be None for Eikonal input gradients")
    if not isinstance(reduction, str) or reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")

    lambda_eikonal = params.pop("lambda_eikonal", 0.1)
    if params:
        raise ValueError(f"unexpected parameters: {', '.join(sorted(params))}")
    if (
        isinstance(lambda_eikonal, bool)
        or not isinstance(lambda_eikonal, numbers.Real)
        or not math.isfinite(float(lambda_eikonal))
        or lambda_eikonal < 0
    ):
        raise ValueError("lambda_eikonal must be a finite nonnegative real number")
    if not isinstance(logits_or_prediction, torch.Tensor) or not logits_or_prediction.is_floating_point():
        raise ValueError("logits_or_prediction must be a floating-point tensor of precomputed input gradients")
    if logits_or_prediction.ndim != 2 or min(logits_or_prediction.shape) < 1:
        raise ValueError("logits_or_prediction must have nonempty shape [B, D]")

    gradient_norms = torch.linalg.vector_norm(logits_or_prediction, ord=2, dim=1)
    losses = float(lambda_eikonal) * (gradient_norms - 1.0).square()
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.mean()
