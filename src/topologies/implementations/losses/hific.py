import math
import numbers

import torch


def hific(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if ignore_index is not None:
        raise ValueError("ignore_index must be None for image distortion")
    if not isinstance(reduction, str) or reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if not isinstance(logits_or_prediction, torch.Tensor) or not logits_or_prediction.is_floating_point():
        raise ValueError("logits_or_prediction must be a floating-point tensor")
    if not isinstance(targets, torch.Tensor) or not targets.is_floating_point():
        raise ValueError("targets must be a floating-point tensor")
    if logits_or_prediction.ndim != 4 or targets.ndim != 4:
        raise ValueError("prediction and targets must have shape [B, 3, H, W]")
    if logits_or_prediction.shape != targets.shape:
        raise ValueError("prediction and targets must have identical shapes")
    if logits_or_prediction.shape[0] < 1 or logits_or_prediction.shape[1] != 3 or min(logits_or_prediction.shape[2:]) < 1:
        raise ValueError("prediction and targets must have nonempty [B, 3, H, W] shapes")
    if logits_or_prediction.dtype != targets.dtype or logits_or_prediction.device != targets.device:
        raise ValueError("prediction and targets must have matching dtype and device")
    if not torch.isfinite(logits_or_prediction).all() or not torch.isfinite(targets).all():
        raise ValueError("prediction and targets must contain only finite values")

    if "lpips_distance" not in params:
        raise ValueError("lpips_distance is required")
    lpips_distance = params.pop("lpips_distance")
    mse_weight = params.pop("mse_weight", 0.00234375)
    lpips_weight = params.pop("lpips_weight", 1.0)
    if params:
        raise ValueError(f"unsupported parameters: {', '.join(sorted(params))}")

    for name, weight in (("mse_weight", mse_weight), ("lpips_weight", lpips_weight)):
        if isinstance(weight, bool) or not isinstance(weight, numbers.Real) or not math.isfinite(float(weight)) or weight < 0:
            raise ValueError(f"{name} must be a finite nonnegative real number")

    batch_size = logits_or_prediction.shape[0]
    try:
        lpips_distance = torch.as_tensor(
            lpips_distance,
            dtype=logits_or_prediction.dtype,
            device=logits_or_prediction.device,
        )
    except (TypeError, RuntimeError):
        raise ValueError("lpips_distance must be a numeric tensor or scalar") from None
    if lpips_distance.ndim == 0:
        if batch_size != 1:
            raise ValueError("scalar lpips_distance is supported only for batch size one")
        lpips_distance = lpips_distance.reshape(1)
    if lpips_distance.shape != (batch_size,):
        raise ValueError("lpips_distance must have shape [B]")
    if not torch.isfinite(lpips_distance).all():
        raise ValueError("lpips_distance must contain only finite values")

    squared_error = (logits_or_prediction - targets).square().flatten(start_dim=1).mean(dim=1)
    per_image = float(mse_weight) * squared_error + float(lpips_weight) * lpips_distance
    if reduction == "none":
        return per_image
    if reduction == "sum":
        return per_image.sum()
    return per_image.mean()
