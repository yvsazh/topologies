import torch


def gram_loss(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    tensors_are_valid = isinstance(logits_or_prediction, torch.Tensor) and isinstance(targets, torch.Tensor)
    if not tensors_are_valid:
        raise ValueError("logits_or_prediction and targets must be torch.Tensor instances")
    if logits_or_prediction.ndim != 4 or targets.ndim != 4:
        raise ValueError("logits_or_prediction and targets must be 4D NCHW tensors")
    if any(size <= 0 for size in logits_or_prediction.shape) or any(size <= 0 for size in targets.shape):
        raise ValueError("logits_or_prediction and targets must have positive dimensions")
    if logits_or_prediction.shape != targets.shape:
        raise ValueError("logits_or_prediction and targets must have identical shapes")
    if not logits_or_prediction.is_floating_point() or not targets.is_floating_point():
        raise ValueError("logits_or_prediction and targets must be floating-point tensors")
    if logits_or_prediction.dtype != targets.dtype:
        raise ValueError("logits_or_prediction and targets must have the same dtype")
    if logits_or_prediction.device != targets.device:
        raise ValueError("logits_or_prediction and targets must be on the same device")
    if ignore_index is not None:
        raise ValueError("ignore_index must be None for feature maps")
    if params:
        raise ValueError(f"unsupported parameters: {', '.join(sorted(params))}")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")

    batch_size, channels, height, width = logits_or_prediction.shape
    positions = height * width
    prediction_features = logits_or_prediction.reshape(batch_size, channels, positions)
    target_features = targets.reshape(batch_size, channels, positions)
    prediction_gram = torch.bmm(prediction_features, prediction_features.transpose(1, 2))
    target_gram = torch.bmm(target_features, target_features.transpose(1, 2))
    squared_distance = (prediction_gram - target_gram).square().sum(dim=(1, 2))
    per_sample = squared_distance / (4 * channels**2 * positions**2)

    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    return per_sample.mean()
