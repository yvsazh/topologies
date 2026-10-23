import math
import numbers

import torch
import torch.nn.functional as F


def fixmatch(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    allowed_params = {"threshold", "unlabeled_weight", "weak_logits", "strong_logits"}
    if set(params) - allowed_params or not {"weak_logits", "strong_logits"}.issubset(params):
        raise ValueError("weak_logits and strong_logits are required and only supported parameters may be supplied")
    threshold = params.get("threshold", 0.95)
    unlabeled_weight = params.get("unlabeled_weight", 1.0)
    weak_logits = params["weak_logits"]
    strong_logits = params["strong_logits"]
    threshold_is_real = not isinstance(threshold, bool) and isinstance(threshold, numbers.Real)
    if not threshold_is_real:
        raise ValueError("threshold must be a finite real number in [0, 1]")
    if not math.isfinite(float(threshold)) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be a finite real number in [0, 1]")
    weight_is_real = not isinstance(unlabeled_weight, bool) and isinstance(unlabeled_weight, numbers.Real)
    if not weight_is_real:
        raise ValueError("unlabeled_weight must be a finite nonnegative real number")
    if not math.isfinite(float(unlabeled_weight)) or unlabeled_weight < 0.0:
        raise ValueError("unlabeled_weight must be a finite nonnegative real number")
    if not isinstance(reduction, str) or reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if ignore_index is not None and (isinstance(ignore_index, bool) or not isinstance(ignore_index, numbers.Integral)):
        raise ValueError("ignore_index must be an integer or None")
    if not isinstance(logits_or_prediction, torch.Tensor) or not logits_or_prediction.is_floating_point():
        raise ValueError("logits_or_prediction must be a floating point tensor")
    if not isinstance(weak_logits, torch.Tensor) or not weak_logits.is_floating_point():
        raise ValueError("weak_logits must be a floating point tensor")
    if not isinstance(strong_logits, torch.Tensor) or not strong_logits.is_floating_point():
        raise ValueError("strong_logits must be a floating point tensor")
    if logits_or_prediction.ndim != 2 or logits_or_prediction.shape[0] < 1 or logits_or_prediction.shape[1] < 1:
        raise ValueError("logits_or_prediction must have shape [B_l, C] with positive dimensions")
    if weak_logits.ndim != 2 or weak_logits.shape[0] < 1 or weak_logits.shape[1] < 1:
        raise ValueError("weak_logits must have shape [B_u, C] with positive dimensions")
    if tuple(strong_logits.shape) != tuple(weak_logits.shape):
        raise ValueError("strong_logits must match weak_logits shape [B_u, C]")
    if weak_logits.shape[1] != logits_or_prediction.shape[1]:
        raise ValueError("all logits must have the same number of classes")
    if not isinstance(targets, torch.Tensor) or targets.dtype != torch.long:
        raise ValueError("targets must be a long tensor of labeled class indices")
    if tuple(targets.shape) != (logits_or_prediction.shape[0],):
        raise ValueError("targets must have shape [B_l]")
    if targets.device != logits_or_prediction.device:
        raise ValueError("targets and labeled logits must be on the same device")
    if weak_logits.device != logits_or_prediction.device or strong_logits.device != logits_or_prediction.device:
        raise ValueError("all logits must be on the same device")
    if weak_logits.dtype != logits_or_prediction.dtype or strong_logits.dtype != logits_or_prediction.dtype:
        raise ValueError("all logits must have the same dtype")
    valid_targets = torch.ones_like(targets, dtype=torch.bool) if ignore_index is None else targets != int(ignore_index)
    class_count = logits_or_prediction.shape[1]
    invalid_targets = valid_targets & ((targets < 0) | (targets >= class_count))
    if bool(invalid_targets.any()):
        raise ValueError("targets contain a class index outside the logits class range")
    safe_targets = targets.masked_fill(~valid_targets, 0)
    labeled_losses = F.cross_entropy(logits_or_prediction, safe_targets, reduction="none")
    labeled_losses = labeled_losses.masked_fill(~valid_targets, 0)
    with torch.no_grad():
        probabilities = weak_logits.softmax(dim=-1)
        confidence, pseudo_labels = probabilities.max(dim=-1)
        accepted = confidence >= float(threshold)
    unlabeled_losses = F.cross_entropy(strong_logits, pseudo_labels, reduction="none")
    unlabeled_losses = unlabeled_losses * accepted.to(dtype=unlabeled_losses.dtype) * float(unlabeled_weight)
    if reduction == "none":
        return torch.cat((labeled_losses, unlabeled_losses))
    if reduction == "sum":
        return labeled_losses.sum() + unlabeled_losses.sum()
    supervised_count = valid_targets.sum().clamp_min(1).to(dtype=labeled_losses.dtype)
    return labeled_losses.sum() / supervised_count + unlabeled_losses.sum() / weak_logits.shape[0]
