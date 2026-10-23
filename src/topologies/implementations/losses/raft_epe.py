import math
import numbers

import torch


def raft_epe(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if ignore_index is not None:
        raise ValueError("ignore_index must be None for optical flow")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if not isinstance(logits_or_prediction, (list, tuple)) or not logits_or_prediction:
        raise ValueError("logits_or_prediction must be a nonempty list or tuple of flow tensors")
    if not isinstance(targets, torch.Tensor) or targets.ndim != 4:
        raise ValueError("targets must be a floating-point tensor with shape [B, 2, H, W]")
    if not targets.is_floating_point() or targets.shape[0] < 1 or targets.shape[1] != 2 or targets.shape[2] < 1 or targets.shape[3] < 1:
        raise ValueError("targets must be a nonempty floating-point tensor with shape [B, 2, H, W]")

    unknown = set(params) - {"valid", "gamma", "max_flow"}
    if unknown:
        raise ValueError(f"unsupported parameters: {sorted(unknown)}")
    if "valid" not in params:
        raise ValueError("valid is required and must have shape [B, H, W]")
    valid = params["valid"]
    if not isinstance(valid, torch.Tensor) or valid.ndim != 3 or tuple(valid.shape) != (targets.shape[0], targets.shape[2], targets.shape[3]):
        raise ValueError("valid must have shape [B, H, W] matching targets")
    if valid.device != targets.device or valid.is_complex():
        raise ValueError("valid must be a real tensor on the same device as targets")

    gamma = params.get("gamma", 0.8)
    if isinstance(gamma, bool) or not isinstance(gamma, numbers.Real) or not math.isfinite(float(gamma)) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be a finite real scalar in [0, 1]")
    max_flow = params.get("max_flow", 400.0)
    if isinstance(max_flow, bool) or not isinstance(max_flow, numbers.Real) or not math.isfinite(float(max_flow)) or max_flow <= 0.0:
        raise ValueError("max_flow must be a finite positive real scalar")

    for prediction in logits_or_prediction:
        if not isinstance(prediction, torch.Tensor) or tuple(prediction.shape) != tuple(targets.shape):
            raise ValueError("every prediction must be a tensor with the same [B, 2, H, W] shape as targets")
        if not prediction.is_floating_point() or prediction.dtype != targets.dtype or prediction.device != targets.device:
            raise ValueError("every prediction must match targets dtype and device and be floating point")

    mask = (valid >= 0.5) & (torch.sqrt(torch.sum(targets.square(), dim=1)) < max_flow)
    mask = mask.unsqueeze(1).to(dtype=targets.dtype)
    loss_map = torch.zeros_like(targets)
    mean_loss = targets.new_zeros(())
    count = len(logits_or_prediction)
    for index, prediction in enumerate(logits_or_prediction):
        weight = gamma ** (count - index - 1)
        component_error = (prediction - targets).abs() * mask
        loss_map = loss_map + weight * component_error
        mean_loss = mean_loss + weight * component_error.mean()

    if reduction == "none":
        return loss_map
    if reduction == "sum":
        return loss_map.sum()
    return mean_loss
