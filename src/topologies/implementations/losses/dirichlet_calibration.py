import math
import numbers

import torch
import torch.nn.functional as F


def dirichlet_calibration(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if set(params) != {"weight", "bias", "lambda", "mu"}:
        raise ValueError("weight, bias, lambda, and mu are required parameters")
    if not isinstance(logits_or_prediction, torch.Tensor) or not logits_or_prediction.is_floating_point():
        raise ValueError("logits_or_prediction must be a floating point tensor")
    if logits_or_prediction.ndim != 2 or logits_or_prediction.shape[1] < 2:
        raise ValueError("logits_or_prediction must have shape [B, K] with at least two classes")
    if not isinstance(targets, torch.Tensor) or (targets.dtype != torch.long and not targets.is_floating_point()):
        raise ValueError("targets must be a long or floating point tensor of class indices")
    if tuple(targets.shape) != (logits_or_prediction.shape[0],):
        raise ValueError("targets must have shape [B] matching logits_or_prediction")
    if targets.device != logits_or_prediction.device:
        raise ValueError("targets and logits_or_prediction must be on the same device")
    if targets.is_floating_point() and (not bool(torch.isfinite(targets).all()) or not bool((targets == targets.round()).all())):
        raise ValueError("floating point targets must contain finite integral class indices")
    if ignore_index is not None and (isinstance(ignore_index, bool) or not isinstance(ignore_index, numbers.Integral)):
        raise ValueError("ignore_index must be an integer or None")
    if not isinstance(reduction, str) or reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")

    weight = params["weight"]
    bias = params["bias"]
    class_count = logits_or_prediction.shape[1]
    if not isinstance(weight, torch.Tensor) or not weight.is_floating_point() or tuple(weight.shape) != (class_count, class_count):
        raise ValueError("weight must be a floating point tensor with shape [K, K]")
    if not isinstance(bias, torch.Tensor) or not bias.is_floating_point() or tuple(bias.shape) != (class_count,):
        raise ValueError("bias must be a floating point tensor with shape [K]")
    if weight.device != logits_or_prediction.device or bias.device != logits_or_prediction.device:
        raise ValueError("weight, bias, and logits_or_prediction must be on the same device")
    if weight.dtype != logits_or_prediction.dtype or bias.dtype != logits_or_prediction.dtype:
        raise ValueError("weight, bias, and logits_or_prediction must have the same dtype")

    strengths = {}
    for name in ("lambda", "mu"):
        value = params[name]
        if isinstance(value, torch.Tensor):
            value_valid = value.ndim == 0 and value.dtype != torch.bool and not value.is_complex()
            if value_valid:
                scalar = float(value.detach().item())
                value_valid = math.isfinite(scalar) and scalar >= 0
                strengths[name] = value.to(device=weight.device, dtype=weight.dtype)
        else:
            value_valid = isinstance(value, numbers.Real) and not isinstance(value, bool)
            if value_valid:
                scalar = float(value)
                value_valid = math.isfinite(scalar) and scalar >= 0
                strengths[name] = scalar
        if not value_valid:
            raise ValueError(f"{name} must be a finite nonnegative scalar")

    target_indices = targets.to(dtype=torch.long)
    if ignore_index is None:
        valid = torch.ones_like(target_indices, dtype=torch.bool)
    else:
        valid = target_indices != int(ignore_index)
    invalid = valid & ((target_indices < 0) | (target_indices >= class_count))
    if bool(invalid.any()):
        raise ValueError("targets contain a class index outside the logits class range")

    safe_targets = target_indices.masked_fill(~valid, 0)
    log_probabilities = F.log_softmax(logits_or_prediction, dim=1)
    calibrated_scores = F.linear(log_probabilities, weight, bias)
    calibrated_log_probabilities = F.log_softmax(calibrated_scores, dim=1)
    losses = -calibrated_log_probabilities.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
    losses = losses.masked_fill(~valid, 0)
    if reduction == "none":
        return losses
    if reduction == "sum":
        reduced = losses.sum()
    else:
        reduced = losses.sum() / valid.sum().to(dtype=losses.dtype)

    off_diagonal_mask = ~torch.eye(class_count, dtype=torch.bool, device=weight.device)
    off_diagonal_penalty = weight.masked_select(off_diagonal_mask).square().sum()
    regularizer = (
        strengths["lambda"] * off_diagonal_penalty / (class_count * (class_count - 1))
        + strengths["mu"] * bias.square().sum() / class_count
    )
    return reduced + regularizer
