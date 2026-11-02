import math
import numbers

import torch


def facenet(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if ignore_index is not None:
        raise ValueError("facenet does not support ignore_index.")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'.")
    if not torch.is_tensor(logits_or_prediction):
        raise ValueError("logits_or_prediction must be a torch.Tensor.")
    if (
        logits_or_prediction.ndim != 3
        or logits_or_prediction.shape[0] == 0
        or logits_or_prediction.shape[1] != 3
        or logits_or_prediction.shape[2] == 0
    ):
        raise ValueError("logits_or_prediction must have shape [T, 3, D] with T and D positive.")
    if not logits_or_prediction.is_floating_point():
        raise ValueError("logits_or_prediction must have a floating-point dtype.")
    margin = params.pop("margin", 0.2)
    if params:
        names = ", ".join(sorted(params))
        raise ValueError(f"unsupported facenet parameter(s): {names}.")
    if (
        isinstance(margin, bool)
        or not isinstance(margin, numbers.Real)
        or not math.isfinite(margin)
        or margin < 0
    ):
        raise ValueError("margin must be a nonnegative finite scalar.")

    anchors = logits_or_prediction[:, 0]
    positives = logits_or_prediction[:, 1]
    negatives = logits_or_prediction[:, 2]
    positive_distance = (anchors - positives).square().sum(dim=-1)
    negative_distance = (anchors - negatives).square().sum(dim=-1)
    losses = (positive_distance - negative_distance + margin).clamp_min(0)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.mean()
