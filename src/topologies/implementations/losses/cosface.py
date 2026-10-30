import math
import numbers

import torch
import torch.nn.functional as F


def cosface(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if set(params) - {"margin", "scale"}:
        raise ValueError("unsupported parameter; only margin and scale are supported")
    if not isinstance(logits_or_prediction, torch.Tensor) or not logits_or_prediction.is_floating_point():
        raise ValueError("logits_or_prediction must be a floating point tensor")
    if logits_or_prediction.ndim != 2 or logits_or_prediction.shape[0] < 1 or logits_or_prediction.shape[1] < 1:
        raise ValueError("logits_or_prediction must have nonempty shape [B, K]")
    if not isinstance(targets, torch.Tensor) or targets.dtype != torch.long:
        raise ValueError("targets must be a long tensor of class indices")
    if tuple(targets.shape) != (logits_or_prediction.shape[0],):
        raise ValueError("targets must have shape [B] matching logits_or_prediction")
    if targets.device != logits_or_prediction.device:
        raise ValueError("targets and logits_or_prediction must be on the same device")
    if ignore_index is not None:
        raise ValueError("ignore_index must be None")
    if not isinstance(reduction, str) or reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    margin = params.get("margin", 0.35)
    margin_valid = isinstance(margin, numbers.Real) and not isinstance(margin, bool)
    if margin_valid:
        margin_valid = math.isfinite(float(margin)) and margin >= 0
    if not margin_valid:
        raise ValueError("margin must be a finite nonnegative real number")
    scale = params.get("scale", 64.0)
    scale_valid = isinstance(scale, numbers.Real) and not isinstance(scale, bool)
    if scale_valid:
        scale_valid = math.isfinite(float(scale)) and scale > 0
    if not scale_valid:
        raise ValueError("scale must be a finite positive real number")
    class_count = logits_or_prediction.shape[1]
    if bool(((targets < 0) | (targets >= class_count)).any()):
        raise ValueError("targets contain a class index outside the cosine score range")
    target_scores = logits_or_prediction.gather(1, targets.unsqueeze(1)) - float(margin)
    margin_scores = logits_or_prediction.scatter(1, targets.unsqueeze(1), target_scores)
    return F.cross_entropy(margin_scores * float(scale), targets, reduction=reduction)
