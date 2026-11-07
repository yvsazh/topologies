import math
import numbers

import torch


def min_snr_gamma(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    unknown = set(params) - {"snr", "gamma", "prediction_type"}
    if unknown:
        raise ValueError(f"unsupported parameters: {sorted(unknown)}")
    if ignore_index is not None:
        raise ValueError("ignore_index must be None for dense regression targets")
    if not isinstance(reduction, str) or reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if not torch.is_tensor(logits_or_prediction) or not torch.is_tensor(targets):
        raise ValueError("prediction and targets must be tensors")
    if not logits_or_prediction.is_floating_point() or not targets.is_floating_point():
        raise ValueError("prediction and targets must be floating-point tensors")
    if logits_or_prediction.ndim < 1 or targets.ndim < 1:
        raise ValueError("prediction and targets must include a batch dimension")
    if logits_or_prediction.shape != targets.shape:
        raise ValueError("prediction and targets must have identical shapes")
    if logits_or_prediction.dtype != targets.dtype or logits_or_prediction.device != targets.device:
        raise ValueError("prediction and targets must have matching dtype and device")
    if logits_or_prediction.shape[0] < 1 or logits_or_prediction[0].numel() < 1:
        raise ValueError("prediction and targets must contain nonempty samples")
    if not torch.isfinite(logits_or_prediction).all() or not torch.isfinite(targets).all():
        raise ValueError("prediction and targets must contain only finite values")

    gamma = params.get("gamma", 5.0)
    if (
        isinstance(gamma, bool)
        or not isinstance(gamma, numbers.Real)
        or not math.isfinite(float(gamma))
        or gamma <= 0
    ):
        raise ValueError("gamma must be positive finite")
    prediction_type = params.get("prediction_type", "x0")
    if not isinstance(prediction_type, str) or prediction_type not in {"x0", "epsilon", "v"}:
        raise ValueError("prediction_type must be 'x0', 'epsilon', or 'v'")
    if "snr" not in params:
        raise ValueError("snr is required as a per-example vector")
    snr = params["snr"]
    if not isinstance(snr, torch.Tensor) or snr.ndim != 1 or snr.shape[0] != logits_or_prediction.shape[0]:
        raise ValueError("snr must be a tensor with shape (batch_size,)")
    if not snr.is_floating_point():
        raise ValueError("snr must contain floating-point values")
    if snr.device != logits_or_prediction.device:
        raise ValueError("snr must be on the prediction device")
    if not torch.isfinite(snr).all() or torch.any(snr < 0):
        raise ValueError("snr must contain nonnegative finite values")

    snr = snr.to(dtype=logits_or_prediction.dtype)
    clipped_snr = snr.clamp(max=float(gamma))
    if prediction_type == "x0":
        weights = clipped_snr
    elif prediction_type == "epsilon":
        zero_snr = snr == 0
        safe_snr = torch.where(zero_snr, torch.ones_like(snr), snr)
        weights = clipped_snr / safe_snr
        weights = torch.where(zero_snr, torch.ones_like(weights), weights)
    else:
        weights = clipped_snr / (snr + 1.0)

    per_sample_mse = (logits_or_prediction - targets).square().reshape(
        logits_or_prediction.shape[0], -1
    ).mean(dim=1)
    per_sample = per_sample_mse * weights
    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    return per_sample.mean()
