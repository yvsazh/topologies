import math

import torch


def crps(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if ignore_index is not None:
        raise ValueError("ignore_index is unsupported for continuous CRPS targets")
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
    if logits_or_prediction.ndim < 1 or logits_or_prediction.shape[-1] != 2:
        raise ValueError("prediction final dimension must contain [mu, sigma]")
    if logits_or_prediction.shape[:-1] != targets.shape:
        raise ValueError("targets must match the prediction leading shape")
    if targets.numel() == 0:
        raise ValueError("prediction and targets must be nonempty")
    if logits_or_prediction.dtype != targets.dtype or logits_or_prediction.device != targets.device:
        raise ValueError("prediction and targets must have matching dtype and device")
    if not torch.isfinite(logits_or_prediction).all() or not torch.isfinite(targets).all():
        raise ValueError("prediction and targets must contain only finite values")

    mean = logits_or_prediction[..., 0]
    scale = logits_or_prediction[..., 1]
    if (scale <= 0).any():
        raise ValueError("sigma must be strictly positive")

    standardized = (targets - mean) / scale
    density = torch.exp(-0.5 * standardized.square()) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + torch.erf(standardized / math.sqrt(2.0)))
    elementwise = scale * (
        standardized * (2.0 * cdf - 1.0)
        + 2.0 * density
        - 1.0 / math.sqrt(math.pi)
    )

    if reduction == "none":
        return elementwise
    if reduction == "sum":
        return elementwise.sum()
    return elementwise.mean()
