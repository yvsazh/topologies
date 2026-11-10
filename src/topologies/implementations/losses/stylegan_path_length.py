import math
import numbers

import torch


def stylegan_path_length(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    def invalid(message):
        raise ValueError(message)

    if ignore_index is not None:
        raise ValueError("ignore_index must be None for path length regularization")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    unsupported = set(params) - {"target_scale", "weight", "noise"}
    if unsupported:
        raise ValueError(f"unsupported parameters: {sorted(unsupported)}")
    if not isinstance(logits_or_prediction, torch.Tensor) or not isinstance(targets, torch.Tensor):
        invalid("logits_or_prediction and targets must be tensors")
    if logits_or_prediction.ndim < 2 or any(size < 1 for size in logits_or_prediction.shape):
        raise ValueError("logits_or_prediction must have nonempty batch and image dimensions")
    if targets.ndim != 2 or any(size < 1 for size in targets.shape):
        raise ValueError("targets must have shape [B,L] with positive dimensions")
    if logits_or_prediction.shape[0] != targets.shape[0]:
        raise ValueError("logits_or_prediction and targets must have the same batch size")
    if not logits_or_prediction.dtype.is_floating_point or not targets.dtype.is_floating_point:
        raise ValueError("logits_or_prediction and targets must have floating-point dtypes")
    if logits_or_prediction.dtype != targets.dtype or logits_or_prediction.device != targets.device:
        raise ValueError("logits_or_prediction and targets must have the same dtype and device")
    if not torch.isfinite(logits_or_prediction).all() or not torch.isfinite(targets).all():
        raise ValueError("logits_or_prediction and targets must contain only finite values")
    if not logits_or_prediction.requires_grad or not targets.requires_grad:
        raise ValueError("logits_or_prediction and targets must require gradients")

    target_scale = params.get("target_scale", 0.0)
    weight = params.get("weight", 1.0)

    def scalar_parameter(value, name):
        if isinstance(value, torch.Tensor):
            if value.numel() != 1 or value.requires_grad:
                raise ValueError(f"{name} must be a detached scalar")
            scalar = value.detach().to(device=logits_or_prediction.device, dtype=logits_or_prediction.dtype).reshape(())
        elif isinstance(value, numbers.Real) and not isinstance(value, bool):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            scalar = logits_or_prediction.new_tensor(value)
        else:
            invalid(f"{name} must be a finite real scalar")
        if not torch.isfinite(scalar):
            raise ValueError(f"{name} must be finite")
        if scalar < 0:
            raise ValueError(f"{name} must be nonnegative")
        return scalar

    target_scale = scalar_parameter(target_scale, "target_scale")
    weight = scalar_parameter(weight, "weight")
    noise = params.get("noise")
    if noise is None:
        noise = torch.randn_like(logits_or_prediction)
    elif not isinstance(noise, torch.Tensor):
        raise ValueError("noise must be a tensor with the same shape, dtype, and device as logits_or_prediction")
    elif (
        noise.shape != logits_or_prediction.shape
        or noise.dtype != logits_or_prediction.dtype
        or noise.device != logits_or_prediction.device
        or not noise.dtype.is_floating_point
        or not torch.isfinite(noise).all()
    ):
        raise ValueError("noise must be finite and have the same shape, dtype, and device as logits_or_prediction")

    path_gradient = torch.autograd.grad(
        (logits_or_prediction * noise).sum(),
        targets,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if path_gradient is None:
        raise ValueError("logits_or_prediction must be differentiably connected to targets")
    lengths = torch.linalg.vector_norm(path_gradient, ord=2, dim=1)
    per_sample = weight * (lengths - target_scale).square()
    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    return per_sample.mean()
