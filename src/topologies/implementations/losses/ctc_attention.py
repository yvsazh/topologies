import math

import torch


def ctc_attention(
    logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params
):
    if ignore_index is not None:
        raise ValueError("ignore_index must be None for sequence-level loss components")
    if "ctc_weight" not in params:
        raise ValueError("ctc_weight is required")
    unexpected = set(params) - {"ctc_weight"}
    if unexpected:
        raise ValueError(f"unexpected parameters: {sorted(unexpected)}")
    ctc_weight = params["ctc_weight"]
    if ctc_weight is True or ctc_weight is False:
        raise ValueError("ctc_weight must be a finite real in [0, 1]")
    try:
        valid_weight = math.isfinite(ctc_weight) and 0 <= ctc_weight <= 1
    except (OverflowError, TypeError, ValueError):
        valid_weight = False
    if not valid_weight:
        raise ValueError("ctc_weight must be a finite real in [0, 1]")
    if not torch.is_tensor(logits_or_prediction) or not torch.is_tensor(targets):
        raise ValueError("loss components must be floating-point tensors with shape [B]")
    if logits_or_prediction.ndim != 1 or targets.ndim != 1:
        raise ValueError("loss components must have shape [B]")
    if logits_or_prediction.shape != targets.shape:
        raise ValueError("loss components must have identical shapes with shape [B]")
    if logits_or_prediction.numel() == 0:
        raise ValueError("loss components must have a nonempty batch")
    if not torch.is_floating_point(logits_or_prediction) or not torch.is_floating_point(
        targets
    ):
        raise ValueError("loss components must be floating-point tensors")
    if logits_or_prediction.dtype != targets.dtype:
        raise ValueError("loss components must have the same dtype")
    if logits_or_prediction.device != targets.device:
        raise ValueError("loss components must be on the same device")
    if not isinstance(reduction, str) or reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    per_sample = ctc_weight * logits_or_prediction + (1 - ctc_weight) * targets
    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    return per_sample.mean()
