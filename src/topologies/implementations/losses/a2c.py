import math

import torch


def a2c(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if ignore_index is not None:
        raise ValueError("ignore_index is not supported")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    unknown = set(params) - {"value", "returns", "entropy", "entropy_coefficient"}
    if unknown:
        raise ValueError(f"unexpected parameters: {sorted(unknown)}")
    missing = {"value", "returns", "entropy"} - set(params)
    if missing:
        raise ValueError(f"missing required parameters: {sorted(missing)}")
    entropy_coefficient = params.get("entropy_coefficient", 0.01)
    if isinstance(entropy_coefficient, bool) or not isinstance(entropy_coefficient, (int, float)):
        error = ValueError("entropy_coefficient must be a finite nonnegative real number")
        raise error
    if not math.isfinite(entropy_coefficient) or entropy_coefficient < 0:
        raise ValueError("entropy_coefficient must be a finite nonnegative real number")
    tensors = {
        "logits_or_prediction": logits_or_prediction,
        "targets": targets,
        "value": params["value"],
        "returns": params["returns"],
        "entropy": params["entropy"],
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            error = ValueError(f"{name} must be a tensor")
            raise error
        if not tensor.is_floating_point():
            raise ValueError(f"{name} must have a floating-point dtype")
        if tensor.ndim != 1 or tensor.shape != logits_or_prediction.shape:
            raise ValueError("all inputs must have identical one-dimensional shapes [B]")
        if tensor.numel() == 0:
            raise ValueError("inputs must contain at least one sample")
        if tensor.device != logits_or_prediction.device:
            raise ValueError("all inputs must be on the same device")
    advantage = targets.detach()
    returns = params["returns"].detach()
    loss = (
        -logits_or_prediction * advantage
        + (returns - params["value"]) ** 2
        - entropy_coefficient * params["entropy"]
    )
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()
