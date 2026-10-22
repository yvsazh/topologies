import torch


def byol_loss(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if params:
        raise ValueError("unsupported parameters")
    if ignore_index is not None:
        raise ValueError("ignore_index must be None for embedding targets")
    if not isinstance(logits_or_prediction, torch.Tensor) or not logits_or_prediction.is_floating_point():
        raise ValueError("logits_or_prediction must be a floating point tensor")
    if not isinstance(targets, torch.Tensor) or not targets.is_floating_point():
        raise ValueError("targets must be a floating point tensor")
    if logits_or_prediction.ndim != 2 or min(logits_or_prediction.shape) < 1:
        raise ValueError("logits_or_prediction must be a non-empty two-dimensional [B, D] tensor")
    if targets.ndim != 2 or min(targets.shape) < 1:
        raise ValueError("targets must be a non-empty two-dimensional [B, D] tensor")
    if logits_or_prediction.shape != targets.shape:
        raise ValueError("logits_or_prediction and targets must have identical shapes")
    if logits_or_prediction.dtype != targets.dtype:
        raise ValueError("logits_or_prediction and targets must have identical dtypes")
    if logits_or_prediction.device != targets.device:
        raise ValueError("logits_or_prediction and targets must be on the same device")
    if not isinstance(reduction, str) or reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")

    detached_targets = targets.detach()
    prediction_norms = torch.linalg.vector_norm(logits_or_prediction, ord=2, dim=1, keepdim=True)
    target_norms = torch.linalg.vector_norm(detached_targets, ord=2, dim=1, keepdim=True)
    if bool((prediction_norms == 0).any()) or bool((target_norms == 0).any()):
        raise ValueError("prediction and target rows must have non-zero norms; zero-norm rows are invalid")

    normalized_prediction = logits_or_prediction / prediction_norms
    normalized_targets = detached_targets / target_norms
    losses = (normalized_prediction - normalized_targets).square().sum(dim=1)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.mean()
