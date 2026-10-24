import math
import numbers

import torch
import torch.nn.functional as F


def asl(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if ignore_index is not None:
        raise ValueError("ignore_index is unsupported for binary ASL targets")
    if not isinstance(reduction, str) or reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")

    unexpected = set(params) - {"gamma_pos", "gamma_neg", "margin"}
    if unexpected:
        raise ValueError(f"unexpected parameters: {', '.join(sorted(unexpected))}")

    gamma_pos = params.get("gamma_pos", 0.0)
    gamma_neg = params.get("gamma_neg", 4.0)
    margin = params.get("margin", 0.05)
    for name, value in (("gamma_pos", gamma_pos), ("gamma_neg", gamma_neg), ("margin", margin)):
        if (
            not isinstance(value, numbers.Real)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value < 0
        ):
            raise ValueError(f"{name} must be a finite nonnegative real number")

    if (
        not isinstance(logits_or_prediction, torch.Tensor)
        or not isinstance(targets, torch.Tensor)
        or not logits_or_prediction.is_floating_point()
    ):
        raise ValueError("logits and targets must be tensors and logits must be floating point")
    if logits_or_prediction.ndim != 2 or targets.shape != logits_or_prediction.shape:
        raise ValueError("logits and targets must have identical two-dimensional shapes [B, K]")
    if logits_or_prediction.shape[0] == 0 or logits_or_prediction.shape[1] == 0:
        raise ValueError("logits and targets must have nonempty batch and label dimensions")
    if targets.is_complex() or targets.device != logits_or_prediction.device:
        raise ValueError("targets must be real-valued and on the same device as logits")
    if not torch.isfinite(targets).all() or ((targets != 0) & (targets != 1)).any():
        raise ValueError("targets must contain only finite binary targets 0 and 1")

    targets = targets.to(dtype=logits_or_prediction.dtype)
    probabilities = torch.sigmoid(logits_or_prediction)
    shifted_negative = (probabilities - float(margin)).clamp_min(0.0)
    log_margin = math.log(float(margin)) if margin > 0 else float("-inf")
    log_negative_complement = torch.logaddexp(
        logits_or_prediction.new_tensor(log_margin), F.logsigmoid(-logits_or_prediction)
    )
    log_negative_complement = torch.where(
        shifted_negative > 0.0,
        log_negative_complement,
        torch.zeros_like(log_negative_complement),
    )
    positive = -targets * (1.0 - probabilities).pow(float(gamma_pos)) * F.logsigmoid(
        logits_or_prediction
    )
    negative = -(1.0 - targets) * shifted_negative.pow(float(gamma_neg)) * log_negative_complement
    per_example = (positive + negative).sum(dim=1)

    if reduction == "none":
        return per_example
    if reduction == "sum":
        return per_example.sum()
    return per_example.mean()
