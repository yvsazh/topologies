import math
import numbers

import torch


def poly1_loss(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if set(params) != {"epsilon"}:
        raise ValueError("epsilon must be supplied and no other parameters are supported")
    epsilon = params["epsilon"]
    if isinstance(epsilon, bool) or not isinstance(epsilon, numbers.Real):
        raise TypeError("epsilon must be a finite real number greater than or equal to -1")
    if not math.isfinite(float(epsilon)) or epsilon < -1:
        raise ValueError("epsilon must be a finite real number greater than or equal to -1")
    if not isinstance(logits_or_prediction, torch.Tensor) or not logits_or_prediction.is_floating_point():
        raise ValueError("logits_or_prediction must be a floating point tensor")
    if logits_or_prediction.ndim < 2 or logits_or_prediction.shape[1] < 1:
        raise ValueError("logits_or_prediction must have shape [B, C, ...] with at least one class")
    if not isinstance(targets, torch.Tensor) or targets.dtype != torch.long:
        raise ValueError("targets must be a long tensor of class indices")
    expected_shape = (logits_or_prediction.shape[0], *logits_or_prediction.shape[2:])
    if tuple(targets.shape) != expected_shape:
        raise ValueError("targets must match logits_or_prediction with the class dimension removed")
    if targets.device != logits_or_prediction.device:
        raise ValueError("targets and logits_or_prediction must be on the same device")
    if ignore_index is not None and (isinstance(ignore_index, bool) or not isinstance(ignore_index, numbers.Integral)):
        raise ValueError("ignore_index must be an integer or None")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    valid = torch.ones_like(targets, dtype=torch.bool) if ignore_index is None else targets != int(ignore_index)
    class_count = logits_or_prediction.shape[1]
    invalid = valid & ((targets < 0) | (targets >= class_count))
    if bool(invalid.any()):
        raise ValueError("targets contain a class index outside the logits class range")
    safe_targets = targets.masked_fill(~valid, 0)
    log_pt = logits_or_prediction.log_softmax(dim=1).gather(1, safe_targets.unsqueeze(1)).squeeze(1)
    losses = -log_pt - float(epsilon) * torch.expm1(log_pt)
    losses = losses.masked_fill(~valid, 0)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.sum() / valid.sum()
