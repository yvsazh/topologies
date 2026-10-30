import torch


def cif_quantity(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if params:
        raise ValueError(f"unsupported parameters: {', '.join(sorted(params))}")
    if ignore_index is not None:
        raise ValueError("ignore_index must be None for CIF quantity loss")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if not isinstance(logits_or_prediction, torch.Tensor) or not torch.is_floating_point(logits_or_prediction):
        raise ValueError("logits_or_prediction must be a floating-point tensor")
    if logits_or_prediction.ndim != 2:
        raise ValueError("logits_or_prediction must have shape [B, U]")
    if logits_or_prediction.shape[0] == 0 or logits_or_prediction.shape[1] == 0:
        raise ValueError("logits_or_prediction must have positive batch and sequence dimensions")
    if not isinstance(targets, torch.Tensor) or targets.ndim != 1:
        raise ValueError("targets must be a rank-1 tensor with shape [B]")
    if targets.shape[0] != logits_or_prediction.shape[0]:
        raise ValueError("targets and logits_or_prediction must have matching batch dimensions")
    if targets.device != logits_or_prediction.device:
        raise ValueError("targets and logits_or_prediction must be on the same device")
    if targets.is_complex() or targets.dtype == torch.bool:
        raise ValueError("targets must contain real numeric target lengths")

    losses = torch.abs(logits_or_prediction.sum(dim=1) - targets.to(dtype=logits_or_prediction.dtype))
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.mean()
