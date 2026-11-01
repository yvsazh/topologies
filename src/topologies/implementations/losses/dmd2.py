import torch


def dmd2(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if params:
        raise ValueError("unsupported parameters")
    if ignore_index is not None:
        raise ValueError("ignore_index must be None for DMD2 score differences")
    if not isinstance(reduction, str) or reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if not isinstance(logits_or_prediction, torch.Tensor) or not logits_or_prediction.is_floating_point():
        raise ValueError("logits_or_prediction must be a floating-point tensor")
    if not isinstance(targets, torch.Tensor) or not targets.is_floating_point():
        raise ValueError("targets must be a floating-point tensor")
    if logits_or_prediction.ndim < 1 or any(size <= 0 for size in logits_or_prediction.shape):
        raise ValueError("logits_or_prediction must have a nonempty batch and event shape [B, ...]")
    if targets.shape != logits_or_prediction.shape:
        raise ValueError("logits_or_prediction and targets must have identical shapes")
    if targets.dtype != logits_or_prediction.dtype:
        raise ValueError("logits_or_prediction and targets must have identical dtypes")
    if targets.device != logits_or_prediction.device:
        raise ValueError("logits_or_prediction and targets must be on the same device")

    batch_size = logits_or_prediction.shape[0]
    per_sample = -(logits_or_prediction * targets.detach()).reshape(batch_size, -1).sum(dim=1)
    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    return per_sample.mean()
