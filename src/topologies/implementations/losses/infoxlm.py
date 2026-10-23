import numbers

import torch


def infoxlm_loss(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if params:
        raise ValueError("unsupported parameters")
    if not isinstance(logits_or_prediction, torch.Tensor) or not logits_or_prediction.is_floating_point():
        raise ValueError("logits_or_prediction must be a floating point tensor")
    if logits_or_prediction.ndim != 2 or logits_or_prediction.shape[0] < 1 or logits_or_prediction.shape[1] < 1:
        raise ValueError("logits_or_prediction must be a non-empty two-dimensional [B, C] tensor")
    if not isinstance(targets, torch.Tensor) or targets.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise ValueError("targets must be an integer tensor of positive candidate indices")
    if targets.ndim != 1 or targets.shape[0] != logits_or_prediction.shape[0]:
        raise ValueError("targets must have shape [B] matching logits_or_prediction")
    if targets.device != logits_or_prediction.device:
        raise ValueError("targets and logits_or_prediction must be on the same device")
    if ignore_index is not None and (isinstance(ignore_index, bool) or not isinstance(ignore_index, numbers.Integral)):
        raise ValueError("ignore_index must be an integer or None")
    if not isinstance(reduction, str) or reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")

    target_indices = targets.to(dtype=torch.long)
    valid = torch.ones_like(target_indices, dtype=torch.bool)
    if ignore_index is not None:
        valid = target_indices != int(ignore_index)
    invalid = valid & ((target_indices < 0) | (target_indices >= logits_or_prediction.shape[1]))
    if bool(invalid.any()):
        raise ValueError("targets contain an index outside the candidate range")

    safe_targets = target_indices.masked_fill(~valid, 0)
    log_probabilities = logits_or_prediction.log_softmax(dim=1)
    losses = -log_probabilities.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
    losses = losses.masked_fill(~valid, 0)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.sum() / valid.sum().clamp_min(1)
