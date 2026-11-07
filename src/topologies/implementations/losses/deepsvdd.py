import torch


def deepsvdd(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if params:
        raise ValueError(f"unsupported parameters: {', '.join(sorted(params))}")
    if ignore_index is not None:
        raise ValueError("ignore_index must be None")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be none, mean, or sum")
    if not torch.is_tensor(logits_or_prediction) or not torch.is_tensor(targets):
        raise ValueError("prediction and targets must be tensors")
    if logits_or_prediction.ndim != 2 or targets.ndim not in {1, 2}:
        raise ValueError("prediction must have shape [B,D] and targets must have shape [D] or [B,D]")
    batch_size, embedding_size = logits_or_prediction.shape
    if batch_size < 1 or embedding_size < 1:
        raise ValueError("prediction must have positive batch and embedding dimensions")
    if targets.ndim == 1:
        if targets.shape[0] != embedding_size:
            raise ValueError("center shape must match the prediction embedding dimension")
        center = targets
    else:
        if targets.shape != (batch_size, embedding_size):
            raise ValueError("batched center shape must match the prediction shape")
        if not torch.equal(targets, targets[:1].expand_as(targets)):
            raise ValueError("batched targets must contain the same fixed center in every row")
        center = targets
    if not torch.is_floating_point(logits_or_prediction) or not torch.is_floating_point(targets):
        raise ValueError("prediction and targets must have floating-point dtypes")
    if logits_or_prediction.dtype != targets.dtype or logits_or_prediction.device != targets.device:
        raise ValueError("prediction and targets must have the same dtype and device")
    if not bool(torch.isfinite(logits_or_prediction).all()) or not bool(torch.isfinite(targets).all()):
        raise ValueError("prediction and targets must contain only finite values")

    distances = (logits_or_prediction - center).square().sum(dim=1)
    if reduction == "none":
        return distances
    if reduction == "sum":
        return distances.sum()
    return distances.mean()
