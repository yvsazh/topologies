import torch


def wasserstein_1d(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if ignore_index is not None:
        raise ValueError("ignore_index must be None for real-valued samples")
    if not isinstance(reduction, str) or reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if params:
        raise ValueError(f"unsupported parameters: {', '.join(sorted(params))}")
    if not isinstance(logits_or_prediction, torch.Tensor):
        exception_type = ValueError
        raise exception_type("logits_or_prediction must be a real-valued tensor")
    if not isinstance(targets, torch.Tensor):
        exception_type = ValueError
        raise exception_type("targets must be a real-valued tensor")
    for values, name in ((logits_or_prediction, "logits_or_prediction"), (targets, "targets")):
        if values.layout != torch.strided or values.is_quantized or values.is_complex() or values.dtype == torch.bool:
            raise ValueError(f"{name} must be a real-valued tensor")
    if logits_or_prediction.ndim != 1 or logits_or_prediction.numel() == 0:
        raise ValueError("logits_or_prediction must have nonempty shape [N]")
    if targets.shape != logits_or_prediction.shape:
        raise ValueError("targets must have shape [N] matching logits_or_prediction")
    if targets.device != logits_or_prediction.device:
        raise ValueError("targets and logits_or_prediction must be on the same device")
    if not bool(torch.isfinite(logits_or_prediction).all()):
        raise ValueError("logits_or_prediction must contain only finite values")
    if not bool(torch.isfinite(targets).all()):
        raise ValueError("targets must contain only finite values")

    sorted_prediction = torch.sort(logits_or_prediction).values
    sorted_targets = torch.sort(targets).values
    if not sorted_prediction.is_floating_point() and not sorted_targets.is_floating_point():
        sorted_prediction = sorted_prediction.to(torch.float64)
        sorted_targets = sorted_targets.to(torch.float64)
    losses = (sorted_prediction - sorted_targets).abs()
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.mean()
