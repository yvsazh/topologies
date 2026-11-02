import math
import numbers

import torch


def dual_focal(
    logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params
):
    if set(params) - {"gamma"}:
        raise ValueError("unsupported parameters; only gamma is supported")

    gamma = params.get("gamma", 0.0)
    gamma_is_real = isinstance(gamma, numbers.Real) and not isinstance(gamma, bool)
    if gamma_is_real:
        try:
            gamma = float(gamma)
        except (OverflowError, TypeError, ValueError):
            gamma_is_real = False
    if not gamma_is_real or not math.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be a finite nonnegative real number")

    if ignore_index is not None and (
        isinstance(ignore_index, bool) or not isinstance(ignore_index, numbers.Integral)
    ):
        raise ValueError("ignore_index must be an integer or None")
    if not isinstance(reduction, str) or reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")

    if not isinstance(logits_or_prediction, torch.Tensor) or not logits_or_prediction.is_floating_point():
        raise ValueError("logits_or_prediction must be a floating-point tensor")
    if logits_or_prediction.ndim < 2 or logits_or_prediction.shape[1] < 2:
        raise ValueError("logits_or_prediction must have shape [N,C,...] with C >= 2")
    if not isinstance(targets, torch.Tensor) or targets.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise ValueError("targets must be an integer tensor of class indices")
    expected_target_shape = (logits_or_prediction.shape[0], *logits_or_prediction.shape[2:])
    if tuple(targets.shape) != expected_target_shape:
        raise ValueError("targets must match the batch and spatial dimensions of logits")
    if logits_or_prediction.numel() == 0:
        raise ValueError("logits_or_prediction and targets must be nonempty")
    if targets.device != logits_or_prediction.device:
        raise ValueError("logits_or_prediction and targets must be on the same device")
    if not bool(torch.isfinite(logits_or_prediction).all()):
        raise ValueError("logits_or_prediction must contain only finite values")

    valid = torch.ones_like(targets, dtype=torch.bool)
    if ignore_index is not None:
        valid = targets != int(ignore_index)
    class_count = logits_or_prediction.shape[1]
    invalid = valid & ((targets < 0) | (targets >= class_count))
    if bool(invalid.any()):
        raise ValueError("targets contain a class index outside the logits class range")

    safe_targets = targets.masked_fill(~valid, 0).long()
    log_probabilities = torch.log_softmax(logits_or_prediction, dim=1)
    selected_log_probabilities = log_probabilities.gather(
        1, safe_targets.unsqueeze(1)
    ).squeeze(1)
    probabilities = log_probabilities.exp()
    selected_probabilities = selected_log_probabilities.exp()
    lower_probabilities = probabilities.masked_fill(
        probabilities >= selected_probabilities.unsqueeze(1), 0
    )
    dual_probabilities = lower_probabilities.max(dim=1).values
    losses = -(
        1 - selected_probabilities + dual_probabilities
    ).pow(gamma) * selected_log_probabilities
    losses = losses.masked_fill(~valid, 0)

    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.sum() / valid.sum().clamp_min(1).to(dtype=losses.dtype)
