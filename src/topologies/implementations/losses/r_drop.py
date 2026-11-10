import math
import numbers

import torch


def r_drop(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if not isinstance(logits_or_prediction, torch.Tensor) or not logits_or_prediction.is_floating_point():
        raise ValueError("logits_or_prediction must be a floating-point tensor")
    if not isinstance(targets, torch.Tensor) or targets.ndim != 1:
        raise ValueError("targets must be a one-dimensional tensor of class indices")
    if logits_or_prediction.ndim != 2 or logits_or_prediction.shape[0] < 2 or logits_or_prediction.shape[1] < 2:
        raise ValueError("logits_or_prediction must have shape [2*B, C] with B positive and C at least two")
    batch_size = logits_or_prediction.shape[0] // 2
    if logits_or_prediction.shape[0] % 2 != 0:
        raise ValueError("logits_or_prediction must contain two equally sized views")
    if targets.shape[0] != batch_size:
        raise ValueError("targets must have shape [B] matching each view")
    if logits_or_prediction.device != targets.device:
        raise ValueError("logits_or_prediction and targets must be on the same device")
    if targets.dtype == torch.bool or targets.is_complex():
        raise ValueError("targets must contain integer class indices")
    if targets.is_floating_point() and (
        not bool(torch.isfinite(targets).all()) or not torch.equal(targets, targets.round())
    ):
        raise ValueError("floating-point targets must be finite integer class indices")
    if not isinstance(reduction, str) or reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if ignore_index is not None and (
        isinstance(ignore_index, bool) or not isinstance(ignore_index, numbers.Integral)
    ):
        raise ValueError("ignore_index must be an integer or None")

    alpha = params.pop("alpha", 1.0)
    if params:
        raise ValueError("unsupported parameters")
    if isinstance(alpha, bool) or not isinstance(alpha, numbers.Real) or not math.isfinite(float(alpha)) or alpha < 0:
        raise ValueError("alpha must be a finite nonnegative real number")
    alpha = float(alpha)

    if not bool(torch.isfinite(logits_or_prediction).all()):
        raise ValueError("logits_or_prediction must contain only finite values")

    class_count = logits_or_prediction.shape[1]
    target_indices = targets.to(dtype=torch.long)
    valid = torch.ones_like(target_indices, dtype=torch.bool)
    if ignore_index is not None:
        valid = target_indices != int(ignore_index)
    if bool((((target_indices < 0) | (target_indices >= class_count)) & valid).any()):
        raise ValueError("targets must contain valid class indices or ignore_index")

    safe_targets = target_indices.clamp(min=0, max=class_count - 1)
    log_probs1 = torch.log_softmax(logits_or_prediction[:batch_size], dim=-1)
    log_probs2 = torch.log_softmax(logits_or_prediction[batch_size:], dim=-1)
    nll1 = -log_probs1.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
    nll2 = -log_probs2.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
    probs1 = log_probs1.exp()
    probs2 = log_probs2.exp()
    kl12 = (probs1 * (log_probs1 - log_probs2)).sum(dim=-1)
    kl21 = (probs2 * (log_probs2 - log_probs1)).sum(dim=-1)
    per_example = nll1 + nll2 + (alpha / 2.0) * (kl12 + kl21)
    per_example = torch.where(valid, per_example, torch.zeros_like(per_example))

    if reduction == "none":
        return per_example
    if reduction == "sum":
        return per_example.sum()
    return per_example.sum() / valid.sum().clamp_min(1)
