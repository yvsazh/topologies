import torch


def generalised_dice(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    value_error = ValueError
    if ignore_index is not None:
        raise ValueError("ignore_index must be None for dense one-hot targets")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if params:
        raise ValueError(f"unsupported parameters: {', '.join(sorted(params))}")
    if not isinstance(logits_or_prediction, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise value_error("prediction and targets must be tensors")
    if logits_or_prediction.layout != torch.strided or targets.layout != torch.strided:
        raise ValueError("prediction and targets must be dense tensors")
    if logits_or_prediction.ndim < 3:
        raise ValueError("prediction and targets must have shape [B, K, *spatial]")
    if logits_or_prediction.shape != targets.shape:
        raise ValueError("prediction and targets must have identical shapes")
    if any(size == 0 for size in logits_or_prediction.shape):
        raise ValueError("prediction and targets dimensions must be nonempty")
    if logits_or_prediction.shape[1] < 2:
        raise ValueError("prediction and targets must contain at least two classes")
    if not torch.is_floating_point(logits_or_prediction):
        raise ValueError("prediction must be floating point")
    if targets.is_complex():
        raise ValueError("targets must contain one-hot values")
    if logits_or_prediction.device != targets.device:
        raise ValueError("prediction and targets must be on the same device")
    if not torch.isfinite(logits_or_prediction).all().item():
        raise ValueError("prediction must contain finite probabilities")
    if ((logits_or_prediction < 0) | (logits_or_prediction > 1)).any().item():
        raise ValueError("prediction probabilities must be in [0, 1]")
    if torch.is_floating_point(targets) and not torch.isfinite(targets).all().item():
        raise ValueError("targets must contain finite one-hot values")
    if ((targets != 0) & (targets != 1)).any().item():
        raise ValueError("targets must contain one-hot values")

    batch_size, class_count = logits_or_prediction.shape[:2]
    calculation_dtype = (
        torch.float32
        if logits_or_prediction.dtype in (torch.float16, torch.bfloat16)
        else logits_or_prediction.dtype
    )
    probabilities = logits_or_prediction.to(dtype=calculation_dtype).reshape(batch_size, class_count, -1)
    labels = targets.to(dtype=calculation_dtype).reshape(batch_size, class_count, -1)
    if (labels.sum(dim=1) != 1).any().item():
        raise ValueError("targets must assign exactly one class to each spatial position")
    volumes = labels.sum(dim=2)
    if (volumes <= 0).any().item():
        raise ValueError("every class must have positive target volume in each sample")

    weights = volumes.square().reciprocal()
    intersections = (probabilities * labels).sum(dim=2)
    numerator = 2 * (weights * intersections).sum(dim=1)
    denominator = (weights * (volumes + probabilities.sum(dim=2))).sum(dim=1)
    losses = 1 - numerator / denominator

    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.mean()
