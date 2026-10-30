import math

import torch


def combo_loss(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    value_error = ValueError
    if ignore_index is not None:
        raise ValueError("ignore_index must be None for dense binary targets")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    unknown = set(params) - {"alpha", "beta"}
    if unknown:
        raise ValueError(f"unsupported parameters: {', '.join(sorted(unknown))}")

    alpha = params.get("alpha", 0.5)
    beta = params.get("beta", 0.5)
    for name, value in (("alpha", alpha), ("beta", beta)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise value_error(f"{name} must be a finite number in [0, 1]")
        if not math.isfinite(value) or value < 0 or value > 1:
            raise ValueError(f"{name} must be a finite number in [0, 1]")

    prediction = logits_or_prediction
    if not isinstance(prediction, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise value_error("prediction and targets must be tensors")
    if prediction.layout != torch.strided or targets.layout != torch.strided:
        raise ValueError("prediction and targets must be dense tensors")
    if prediction.ndim < 3:
        raise ValueError("prediction and targets must have shape [B, C, *spatial]")
    if prediction.shape != targets.shape:
        raise ValueError("prediction and targets must have identical shapes")
    if any(size == 0 for size in prediction.shape):
        raise ValueError("prediction and targets dimensions must be nonempty")
    if not torch.is_floating_point(prediction):
        raise ValueError("prediction must be floating point")
    if targets.is_complex():
        raise ValueError("targets must contain binary values")
    if prediction.device != targets.device:
        raise ValueError("prediction and targets must be on the same device")
    if not torch.isfinite(prediction).all().item():
        raise ValueError("prediction must contain finite probabilities")
    if ((prediction < 0) | (prediction > 1)).any().item():
        raise ValueError("prediction probabilities must be in [0, 1]")
    if torch.is_floating_point(targets) and not torch.isfinite(targets).all().item():
        raise ValueError("targets must contain finite binary values")
    if ((targets != 0) & (targets != 1)).any().item():
        raise ValueError("targets must contain binary values 0 or 1")

    batch_size, class_count = prediction.shape[:2]
    probabilities = prediction.reshape(batch_size, class_count, -1)
    labels = targets.to(dtype=prediction.dtype).reshape(batch_size, class_count, -1)
    weights = beta * labels + (1 - beta) * (1 - labels)
    cross_entropy = torch.nn.functional.binary_cross_entropy(
        probabilities, labels, weight=weights, reduction="none"
    ).reshape(batch_size, -1).mean(dim=1)
    intersections = (probabilities * labels).sum(dim=2)
    dice_scores = (2 * intersections + 1) / (
        probabilities.sum(dim=2) + labels.sum(dim=2) + 1
    )
    loss = alpha * cross_entropy - (1 - alpha) * dice_scores.sum(dim=1)

    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()
