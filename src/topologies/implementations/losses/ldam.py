import math
import numbers

import torch


def ldam(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if set(params) - {"counts", "max_margin"}:
        raise ValueError("unsupported parameter; only counts and max_margin are supported")
    if "counts" not in params:
        raise ValueError("counts must be provided with one positive value per class")
    if not isinstance(logits_or_prediction, torch.Tensor) or not logits_or_prediction.is_floating_point():
        raise ValueError("logits_or_prediction must be a floating point tensor")
    if logits_or_prediction.ndim != 2 or logits_or_prediction.shape[0] < 1 or logits_or_prediction.shape[1] < 1:
        raise ValueError("logits_or_prediction must have shape [B, K] with positive dimensions")
    if not isinstance(targets, torch.Tensor) or targets.dtype != torch.long:
        raise ValueError("targets must be a long tensor of class indices")
    if tuple(targets.shape) != (logits_or_prediction.shape[0],):
        raise ValueError("targets must have shape [B] matching logits_or_prediction")
    if targets.device != logits_or_prediction.device:
        raise ValueError("targets and logits_or_prediction must be on the same device")
    if ignore_index is not None and (isinstance(ignore_index, bool) or not isinstance(ignore_index, numbers.Integral)):
        raise ValueError("ignore_index must be an integer or None")
    if not isinstance(reduction, str) or reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    max_margin = params.get("max_margin", 0.5)
    if isinstance(max_margin, torch.Tensor):
        margin_valid = max_margin.ndim == 0 and max_margin.dtype != torch.bool and not max_margin.is_complex()
        if margin_valid:
            margin_value = max_margin.detach().item()
            margin_valid = math.isfinite(float(margin_value)) and margin_value > 0
    else:
        margin_valid = isinstance(max_margin, numbers.Real) and not isinstance(max_margin, bool)
        if margin_valid:
            margin_valid = math.isfinite(float(max_margin)) and max_margin > 0
    if not margin_valid:
        raise ValueError("max_margin must be a positive finite scalar")
    try:
        class_counts = torch.as_tensor(params["counts"])
    except (TypeError, ValueError, RuntimeError):
        raise ValueError("counts must be a finite positive vector with one value per class") from None
    if class_counts.ndim != 1 or class_counts.shape[0] != logits_or_prediction.shape[1]:
        raise ValueError("counts must have shape [K] matching the logits class count")
    if class_counts.dtype == torch.bool or class_counts.is_complex():
        raise ValueError("counts must contain finite positive real values")
    class_counts = class_counts.detach().to(device=logits_or_prediction.device, dtype=torch.float64)
    if not bool(torch.isfinite(class_counts).all()) or not bool((class_counts > 0).all()):
        raise ValueError("counts must contain finite positive real values")
    if ignore_index is None:
        valid = torch.ones_like(targets, dtype=torch.bool)
    else:
        valid = targets != int(ignore_index)
    class_count = logits_or_prediction.shape[1]
    invalid = valid & ((targets < 0) | (targets >= class_count))
    if bool(invalid.any()):
        raise ValueError("targets contain a class index outside the logits class range")
    margins = (class_counts.min() / class_counts).pow(0.25) * float(margin_value if isinstance(max_margin, torch.Tensor) else max_margin)
    margins = margins.to(dtype=logits_or_prediction.dtype)
    safe_targets = targets.masked_fill(~valid, 0)
    target_offsets = torch.nn.functional.one_hot(safe_targets, num_classes=class_count).to(dtype=logits_or_prediction.dtype) * margins
    losses = torch.nn.functional.cross_entropy(logits_or_prediction - target_offsets, safe_targets, reduction="none")
    losses = losses.masked_fill(~valid, 0)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.sum() / valid.sum().to(dtype=losses.dtype)
