import math
import numbers

import torch


def evidential(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if not isinstance(logits_or_prediction, torch.Tensor) or logits_or_prediction.ndim != 2:
        raise ValueError("evidence must be a tensor with shape [B, K]")
    if not logits_or_prediction.is_floating_point():
        raise ValueError("evidence must have a floating point dtype")
    if logits_or_prediction.shape[0] == 0 or logits_or_prediction.shape[1] == 0:
        raise ValueError("evidence must have nonempty batch and class dimensions")
    if not torch.isfinite(logits_or_prediction).all():
        raise ValueError("evidence must contain only finite values")
    if (logits_or_prediction < 0).any():
        raise ValueError("evidence values must be nonnegative")
    if not isinstance(targets, torch.Tensor) or targets.ndim != 1:
        raise ValueError("targets must have shape [B]")
    if targets.shape[0] != logits_or_prediction.shape[0]:
        raise ValueError("targets must have shape [B] matching the evidence batch")
    if targets.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("targets must use an integer dtype")
    if ignore_index is not None:
        raise ValueError("ignore_index must be None")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be none, mean, or sum")
    annealing_coef = params.pop("annealing_coef", 1.0)
    if params:
        raise ValueError(f"unsupported parameter: {next(iter(params))}")
    if isinstance(annealing_coef, bool) or not isinstance(annealing_coef, numbers.Real):
        error = ValueError("annealing_coef must be a finite real value in [0, 1]")
        raise error
    if not math.isfinite(annealing_coef) or not 0.0 <= annealing_coef <= 1.0:
        raise ValueError("annealing_coef must be a finite real value in [0, 1]")
    class_count = logits_or_prediction.shape[1]
    if (targets < 0).any() or (targets >= class_count).any():
        raise ValueError("targets must be class indices in range [0, K)")

    alpha = logits_or_prediction + 1.0
    strength = alpha.sum(dim=1, keepdim=True)
    mean = alpha / strength
    variance = alpha * (strength - alpha) / (strength.square() * (strength + 1.0))
    one_hot = torch.zeros_like(alpha).scatter_(1, targets.to(device=alpha.device, dtype=torch.long).unsqueeze(1), 1.0)
    mse = ((one_hot - mean).square() + variance).sum(dim=1)

    adjusted_alpha = one_hot + (1.0 - one_hot) * alpha
    adjusted_strength = adjusted_alpha.sum(dim=1, keepdim=True)
    kl = (
        torch.lgamma(adjusted_strength.squeeze(1))
        - torch.lgamma(torch.as_tensor(class_count, dtype=alpha.dtype, device=alpha.device))
        - torch.lgamma(adjusted_alpha).sum(dim=1)
        + ((adjusted_alpha - 1.0) * (torch.digamma(adjusted_alpha) - torch.digamma(adjusted_strength))).sum(dim=1)
    )
    losses = mse + float(annealing_coef) * kl
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.mean()
