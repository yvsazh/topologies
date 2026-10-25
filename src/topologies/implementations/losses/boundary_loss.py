import torch


def boundary_loss(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if params:
        raise ValueError(f"unsupported parameters: {sorted(params)}")
    if ignore_index is not None:
        raise ValueError("ignore_index must be None for continuous signed distance targets")
    if not isinstance(reduction, str) or reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if not isinstance(logits_or_prediction, torch.Tensor) or not logits_or_prediction.is_floating_point():
        raise ValueError("logits_or_prediction must be a floating point probability tensor")
    if logits_or_prediction.ndim < 3 or logits_or_prediction.shape[1] != 1:
        raise ValueError("logits_or_prediction must have shape [B, 1, *spatial]")
    if any(size < 1 for size in logits_or_prediction.shape):
        raise ValueError("logits_or_prediction dimensions must be nonempty")
    if not isinstance(targets, torch.Tensor) or not targets.is_floating_point():
        raise ValueError("targets must be a floating point signed distance tensor")
    if tuple(targets.shape) != tuple(logits_or_prediction.shape):
        raise ValueError("targets and logits_or_prediction must have the same shape")
    if targets.device != logits_or_prediction.device or targets.dtype != logits_or_prediction.dtype:
        raise ValueError("targets and logits_or_prediction must have the same dtype and device")
    if not bool(torch.isfinite(logits_or_prediction).all()):
        raise ValueError("probabilities must contain only finite values")
    if bool(((logits_or_prediction < 0) | (logits_or_prediction > 1)).any()):
        raise ValueError("probabilities must be in [0, 1]")
    if not bool(torch.isfinite(targets).all()):
        raise ValueError("signed distance values must be finite")

    loss_map = logits_or_prediction * targets
    if reduction == "none":
        return loss_map
    if reduction == "sum":
        return loss_map.sum()
    return loss_map.mean()
