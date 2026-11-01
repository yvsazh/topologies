import math
import numbers

import torch


def si_log(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    unknown = set(params) - {"lambda_"}
    if unknown:
        raise ValueError(f"unsupported parameters: {sorted(unknown)}")
    if not isinstance(reduction, str) or reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if any(not isinstance(value, torch.Tensor) for value in (logits_or_prediction, targets)):
        raise ValueError("prediction and targets must be tensors")
    if not logits_or_prediction.is_floating_point() or not targets.is_floating_point():
        raise ValueError("prediction and targets must be floating-point tensors")
    if logits_or_prediction.ndim < 1 or targets.ndim < 1:
        raise ValueError("prediction and targets must include a batch dimension")
    if logits_or_prediction.shape != targets.shape:
        raise ValueError("prediction and targets must have identical shapes")
    if logits_or_prediction.dtype != targets.dtype or logits_or_prediction.device != targets.device:
        raise ValueError("prediction and targets must have matching dtype and device")
    if logits_or_prediction.shape[0] < 1 or logits_or_prediction[0].numel() < 1:
        raise ValueError("prediction and targets must contain nonempty samples")

    lambda_ = params.get("lambda_", 0.5)
    if isinstance(lambda_, bool) or not isinstance(lambda_, numbers.Real) or not math.isfinite(float(lambda_)) or not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda_ must be a finite real scalar in [0, 1]")
    if ignore_index is not None and (
        isinstance(ignore_index, bool)
        or not isinstance(ignore_index, numbers.Real)
        or not math.isfinite(float(ignore_index))
    ):
        raise ValueError("ignore_index must be None or a finite real target value")

    mask = torch.ones_like(targets, dtype=torch.bool) if ignore_index is None else targets != ignore_index
    flat_mask = mask.reshape(targets.shape[0], -1)
    counts = flat_mask.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("every sample must contain at least one valid target")
    valid_targets = targets[mask]
    if not torch.isfinite(valid_targets).all() or torch.any(valid_targets <= 0):
        raise ValueError("valid targets must contain only positive finite depths")
    valid_predictions = logits_or_prediction[mask]
    if not torch.isfinite(valid_predictions).all():
        raise ValueError("predictions at valid targets must be finite log depths")

    safe_targets = torch.where(mask, targets, torch.ones_like(targets))
    safe_predictions = torch.where(mask, logits_or_prediction, torch.zeros_like(logits_or_prediction))
    residuals = (safe_predictions - safe_targets.log()).reshape(targets.shape[0], -1)
    valid_counts = counts.to(dtype=logits_or_prediction.dtype)
    total = residuals.sum(dim=1)
    squared_total = residuals.square().sum(dim=1)
    per_sample = squared_total / valid_counts - float(lambda_) * total.square() / valid_counts.square()

    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    return per_sample.mean()
