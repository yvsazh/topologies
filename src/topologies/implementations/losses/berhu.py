import torch


def berhu(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if ignore_index is not None:
        raise ValueError("ignore_index is unsupported for BerHu regression targets")
    if not isinstance(reduction, str) or reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if params:
        raise ValueError(f"unexpected parameters: {', '.join(sorted(params))}")
    if (
        not isinstance(logits_or_prediction, torch.Tensor)
        or not isinstance(targets, torch.Tensor)
        or not logits_or_prediction.is_floating_point()
        or not targets.is_floating_point()
    ):
        raise ValueError("prediction and targets must be floating-point tensors")
    if logits_or_prediction.shape != targets.shape:
        raise ValueError("prediction and targets must have identical shapes")
    if logits_or_prediction.numel() == 0:
        raise ValueError("prediction and targets must be nonempty")
    if logits_or_prediction.dtype != targets.dtype or logits_or_prediction.device != targets.device:
        raise ValueError("prediction and targets must have matching dtype and device")
    if not torch.isfinite(logits_or_prediction).all() or not torch.isfinite(targets).all():
        raise ValueError("prediction and targets must contain only finite values")

    residual = logits_or_prediction - targets
    absolute_residual = residual.abs()
    threshold = absolute_residual.amax() / 5.0
    safe_threshold = torch.where(threshold > 0, threshold, torch.ones_like(threshold))
    quadratic = (residual.square() + safe_threshold.square()) / (2.0 * safe_threshold)
    elementwise = torch.where(absolute_residual <= threshold, absolute_residual, quadratic)

    if reduction == "none":
        return elementwise
    if reduction == "sum":
        return elementwise.sum()
    return elementwise.mean()
