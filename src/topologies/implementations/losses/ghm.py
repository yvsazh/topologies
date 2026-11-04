import numbers

import torch
import torch.nn.functional as F


def ghm(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    unexpected = set(params) - {"bins"}
    if unexpected:
        raise ValueError(f"unexpected parameters: {', '.join(sorted(unexpected))}")

    bins = params.get("bins", 30)
    if not isinstance(bins, numbers.Integral) or isinstance(bins, bool) or bins < 1:
        raise ValueError("bins must be a positive integer")
    bins = int(bins)
    if not isinstance(reduction, str) or reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if not torch.is_tensor(logits_or_prediction) or not torch.is_tensor(targets):
        raise ValueError("logits and targets must be tensors")
    if not logits_or_prediction.is_floating_point():
        raise ValueError("logits must be floating point")
    if logits_or_prediction.shape != targets.shape:
        raise ValueError("logits and targets must have identical shapes")
    if logits_or_prediction.numel() == 0:
        raise ValueError("logits and targets must be nonempty")
    if targets.is_complex() or targets.device != logits_or_prediction.device:
        raise ValueError("targets must be real-valued and on the same device as logits")
    if not torch.isfinite(logits_or_prediction).all():
        raise ValueError("logits must contain only finite values")

    if ignore_index is None:
        valid = torch.ones_like(targets, dtype=torch.bool)
    else:
        valid = targets != ignore_index
    valid_targets = targets[valid]
    if valid_targets.numel() == 0:
        raise ValueError("all elements are ignored")
    if not torch.isfinite(valid_targets).all() or ((valid_targets != 0) & (valid_targets != 1)).any():
        raise ValueError("targets must contain only finite binary values 0 and 1 outside ignore_index")

    valid_logits = logits_or_prediction[valid]
    valid_targets = valid_targets.to(dtype=logits_or_prediction.dtype)
    gradient_norms = (torch.sigmoid(valid_logits) - valid_targets).abs()
    bin_indices = torch.floor(gradient_norms * bins).to(dtype=torch.long).clamp(max=bins - 1)
    bin_counts = torch.bincount(bin_indices, minlength=bins)
    densities = bin_counts[bin_indices].to(dtype=logits_or_prediction.dtype) * bins
    weights = valid_logits.new_tensor(valid_logits.numel()) / densities
    element_losses = F.binary_cross_entropy_with_logits(valid_logits, valid_targets, reduction="none")
    weighted_losses = weights * element_losses

    if reduction == "none":
        result = logits_or_prediction.new_zeros(logits_or_prediction.numel())
        result[valid.reshape(-1)] = weighted_losses
        return result.reshape_as(logits_or_prediction)
    if reduction == "sum":
        return weighted_losses.sum()
    return weighted_losses.sum() / valid_logits.numel()
