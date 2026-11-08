import math
import numbers

import torch


def swav_loss(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if ignore_index is not None:
        raise ValueError("ignore_index must be None for soft assignment codes")
    if not isinstance(reduction, str) or reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    temperature = params.pop("temperature", 0.1)
    if params:
        raise ValueError(f"unsupported parameters: {', '.join(sorted(params))}")
    if isinstance(temperature, bool) or not isinstance(temperature, numbers.Real):
        temperature = float("nan")
    else:
        try:
            temperature = float(temperature)
        except OverflowError:
            temperature = float("inf")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a finite positive real number")
    if not isinstance(logits_or_prediction, torch.Tensor) or not logits_or_prediction.is_floating_point():
        raise ValueError("logits_or_prediction must be a floating-point tensor")
    if not isinstance(targets, torch.Tensor) or not targets.is_floating_point():
        raise ValueError("targets must be a floating-point tensor")
    if (
        logits_or_prediction.ndim != 3
        or logits_or_prediction.shape[0] < 1
        or logits_or_prediction.shape[1] != 2
        or logits_or_prediction.shape[2] < 2
    ):
        raise ValueError("logits_or_prediction must have shape [B,2,K] with B > 0 and K >= 2")
    if targets.shape != logits_or_prediction.shape:
        raise ValueError("targets must match logits_or_prediction shape [B,2,K]")
    if logits_or_prediction.dtype != targets.dtype or logits_or_prediction.device != targets.device:
        raise ValueError("logits_or_prediction and targets must share dtype and device")
    if not torch.isfinite(logits_or_prediction).all().item() or not torch.isfinite(targets).all().item():
        raise ValueError("logits_or_prediction and targets must be finite")
    if (targets < 0).any().item() or (targets > 1).any().item():
        raise ValueError("targets must contain valid probability codes in [0,1]")
    codes = targets.detach()
    code_sums = codes.sum(dim=-1)
    if not torch.allclose(code_sums, torch.ones_like(code_sums), rtol=1e-4, atol=1e-5):
        raise ValueError("each target view must contain a valid probability code summing to one")
    log_probabilities = torch.log_softmax(logits_or_prediction / temperature, dim=-1)
    losses = -0.5 * (
        (codes[:, 0] * log_probabilities[:, 1]).sum(dim=-1)
        + (codes[:, 1] * log_probabilities[:, 0]).sum(dim=-1)
    )
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.mean()
