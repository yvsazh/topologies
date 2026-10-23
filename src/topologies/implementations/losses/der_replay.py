import math
from numbers import Real

import torch


def der_replay(
    logits_or_prediction,
    targets,
    *,
    ignore_index=None,
    reduction="mean",
    **params,
):
    def raise_value_error(message):
        raise ValueError(message)

    if ignore_index is not None:
        raise ValueError("ignore_index must be None for stored-logit targets")
    if not isinstance(logits_or_prediction, torch.Tensor):
        raise_value_error("logits_or_prediction must be a tensor")
    if not isinstance(targets, torch.Tensor):
        raise_value_error("targets must be a tensor")
    if logits_or_prediction.ndim != 2 or targets.ndim != 2:
        raise ValueError("logits_or_prediction and targets must have shape [B, C]")
    if logits_or_prediction.shape != targets.shape:
        raise ValueError("logits_or_prediction and targets must have identical shapes")
    if logits_or_prediction.shape[0] == 0 or logits_or_prediction.shape[1] == 0:
        raise ValueError("logit batches and class dimensions must be non-empty")
    if not torch.is_floating_point(logits_or_prediction) or not torch.is_floating_point(targets):
        raise ValueError("logits_or_prediction and targets must use floating-point dtypes")
    if logits_or_prediction.dtype != targets.dtype:
        raise ValueError("logits_or_prediction and targets must have identical dtypes")
    if logits_or_prediction.device != targets.device:
        raise ValueError("logits_or_prediction and targets must be on the same device")
    if set(params) - {"alpha"}:
        raise ValueError("unsupported parameters: " + ", ".join(sorted(set(params) - {"alpha"})))
    alpha = params.get("alpha", 0.5)
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise_value_error("alpha must be a finite nonnegative real number")
    try:
        alpha = float(alpha)
    except OverflowError as exc:
        raise ValueError("alpha must be a finite nonnegative real number") from exc
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be a finite nonnegative real number")
    if not isinstance(reduction, str) or reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")

    per_sample = (logits_or_prediction - targets.detach()).square().sum(dim=1) * alpha
    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    return per_sample.mean()
