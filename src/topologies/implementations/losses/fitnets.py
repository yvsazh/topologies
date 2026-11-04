import torch


def fitnets(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if ignore_index is not None:
        raise ValueError("ignore_index must be None for activation targets")
    if not isinstance(logits_or_prediction, torch.Tensor) or not logits_or_prediction.is_floating_point():
        raise ValueError("logits_or_prediction must be a floating-point tensor")
    if not isinstance(targets, torch.Tensor) or not targets.is_floating_point():
        raise ValueError("targets must be a floating-point tensor")
    if logits_or_prediction.ndim < 2 or any(size < 1 for size in logits_or_prediction.shape):
        raise ValueError("logits_or_prediction must have a batch dimension and non-empty activation dimensions")
    if targets.ndim < 2 or any(size < 1 for size in targets.shape):
        raise ValueError("targets must have a batch dimension and non-empty activation dimensions")
    if logits_or_prediction.shape != targets.shape:
        raise ValueError("logits_or_prediction and targets must have matching shapes")
    if logits_or_prediction.dtype != targets.dtype:
        raise ValueError("logits_or_prediction and targets must have matching dtypes")
    if logits_or_prediction.device != targets.device:
        raise ValueError("logits_or_prediction and targets must be on the same device")
    if not isinstance(reduction, str) or reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if params:
        raise ValueError("unsupported parameters")

    losses = 0.5 * (logits_or_prediction - targets.detach()).square().flatten(start_dim=1).sum(dim=1)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.mean()
