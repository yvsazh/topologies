import math
from numbers import Real

import torch


def dqn_huber(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if ignore_index is not None:
        raise ValueError("ignore_index is unsupported for DQN Huber regression targets")
    if not isinstance(reduction, str) or reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    unexpected = set(params) - {"delta"}
    if unexpected:
        raise ValueError(f"unexpected parameters: {', '.join(sorted(unexpected))}")
    delta = params.get("delta", 1.0)
    valid_delta_type = isinstance(delta, Real) and not isinstance(delta, bool)
    try:
        delta = float(delta)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("delta must be a finite positive real scalar") from error
    if not valid_delta_type or not math.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be a finite positive real scalar")
    if (
        not isinstance(logits_or_prediction, torch.Tensor)
        or not isinstance(targets, torch.Tensor)
        or not logits_or_prediction.is_floating_point()
        or not targets.is_floating_point()
    ):
        raise ValueError("prediction and targets must be floating-point tensors")
    if logits_or_prediction.shape != targets.shape:
        raise ValueError("prediction and targets must have identical shapes")
    if logits_or_prediction.numel() == 0:
        raise ValueError("prediction and targets must be nonempty")
    if logits_or_prediction.dtype != targets.dtype or logits_or_prediction.device != targets.device:
        raise ValueError("prediction and targets must have matching dtype and device")
    if not torch.isfinite(logits_or_prediction).all() or not torch.isfinite(targets).all():
        raise ValueError("prediction and targets must contain only finite values")

    td_error = logits_or_prediction - targets.detach()
    absolute_error = td_error.abs()
    elementwise = torch.where(
        absolute_error <= delta,
        0.5 * td_error.square(),
        delta * (absolute_error - 0.5 * delta),
    )
    if reduction == "none":
        return elementwise
    if reduction == "sum":
        return elementwise.sum()
    return elementwise.mean()
