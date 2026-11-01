import math
import numbers

import torch
import torch.nn.functional as F


def quality_focal_loss(
    logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params
):
    if ignore_index is not None:
        raise ValueError("ignore_index is unsupported for continuous quality targets")
    if not isinstance(reduction, str) or reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")

    unexpected = set(params) - {"beta", "avg_factor"}
    if unexpected:
        raise ValueError(f"unexpected parameters: {', '.join(sorted(unexpected))}")

    beta = params.get("beta", 2.0)
    valid_beta = isinstance(beta, numbers.Real) and not isinstance(beta, bool)
    if valid_beta:
        try:
            valid_beta = math.isfinite(beta) and beta >= 0
        except (OverflowError, TypeError, ValueError):
            valid_beta = False
    if not valid_beta:
        raise ValueError("beta must be a finite nonnegative real number")
    beta = float(beta)

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
        raise ValueError("logits and targets must be floating-point tensors")
    if logits_or_prediction.shape != targets.shape:
        raise ValueError("logits and targets must have identical shapes")
    if logits_or_prediction.numel() == 0:
        raise ValueError("logits and targets must be nonempty")
    if logits_or_prediction.device != targets.device:
        raise ValueError("logits and targets must be on the same device")
    if not torch.isfinite(logits_or_prediction).all():
        raise ValueError("logits must contain only finite values")
    if not torch.isfinite(targets).all() or ((targets < 0) | (targets > 1)).any():
        raise ValueError("targets must contain finite values in [0, 1]")

    targets = targets.to(dtype=logits_or_prediction.dtype)
    if beta == 0.0:
        modulation = torch.ones_like(logits_or_prediction)
    else:
        modulation = (targets - logits_or_prediction.sigmoid()).abs().pow(beta)
    elementwise = modulation * F.binary_cross_entropy_with_logits(
        logits_or_prediction, targets, reduction="none"
    )

    if reduction == "none":
        return elementwise
    if reduction == "sum":
        return elementwise.sum()
    if avg_factor is not None:
        return elementwise.sum() / avg_factor
    return elementwise.mean()
