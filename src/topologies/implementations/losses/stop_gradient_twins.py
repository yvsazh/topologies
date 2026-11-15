import torch


def stop_gradient_twins(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if params:
        raise ValueError(f"unexpected parameters: {', '.join(sorted(params))}")
    if ignore_index is not None:
        raise ValueError("ignore_index is not supported")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if not isinstance(logits_or_prediction, torch.Tensor) or not logits_or_prediction.is_floating_point():
        raise ValueError("logits_or_prediction must be a floating-point tensor")
    if logits_or_prediction.ndim != 2:
        raise ValueError("logits_or_prediction must have shape [B, D]")
    batch_size, feature_count = logits_or_prediction.shape
    if batch_size < 1 or feature_count < 1:
        raise ValueError("logits_or_prediction must have B >= 1 and D >= 1")
    if not isinstance(targets, torch.Tensor) or not targets.is_floating_point():
        raise ValueError("targets must be a floating-point tensor")
    if targets.shape != logits_or_prediction.shape:
        raise ValueError("targets must have the same shape as logits_or_prediction")
    if targets.dtype != logits_or_prediction.dtype:
        raise ValueError("targets and logits_or_prediction must have the same dtype")
    if targets.device != logits_or_prediction.device:
        raise ValueError("targets and logits_or_prediction must be on the same device")
    if not torch.isfinite(logits_or_prediction).all():
        raise ValueError("logits_or_prediction must contain only finite values")
    if not torch.isfinite(targets).all():
        raise ValueError("targets must contain only finite values")

    detached_targets = targets.detach()
    prediction_norms = torch.linalg.vector_norm(logits_or_prediction, ord=2, dim=-1, keepdim=True)
    target_norms = torch.linalg.vector_norm(detached_targets, ord=2, dim=-1, keepdim=True)
    if (prediction_norms == 0).any():
        raise ValueError("prediction vectors must have nonzero norm")
    if (target_norms == 0).any():
        raise ValueError("target vectors must have nonzero norm")

    normalized_predictions = logits_or_prediction / prediction_norms
    normalized_targets = detached_targets / target_norms
    per_example = -(normalized_predictions * normalized_targets).sum(dim=-1)
    if reduction == "none":
        return per_example
    if reduction == "sum":
        return per_example.sum()
    return per_example.mean()
