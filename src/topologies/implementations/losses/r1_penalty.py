import math
import numbers

import torch


def r1_penalty(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    def invalid_value(message):
        raise ValueError(message)

    if params.keys() - {"gamma"}:
        raise ValueError("unsupported parameters")
    if ignore_index is not None:
        raise ValueError("ignore_index must be None")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")

    gamma = params.get("gamma", 10.0)
    if isinstance(gamma, bool) or not isinstance(gamma, numbers.Real):
        invalid_value("gamma must be a finite nonnegative real scalar")
    try:
        gamma = float(gamma)
    except OverflowError as exc:
        raise ValueError("gamma must be a finite nonnegative real scalar") from exc
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be a finite nonnegative real scalar")

    if not isinstance(logits_or_prediction, torch.Tensor) or not isinstance(targets, torch.Tensor):
        invalid_value("samples and discriminator scores must be tensors")
    if logits_or_prediction.ndim < 1 or logits_or_prediction.shape[0] == 0:
        raise ValueError("samples must have a nonempty batch dimension")
    if not logits_or_prediction.is_floating_point():
        raise ValueError("samples must be floating-point tensors")
    if not logits_or_prediction.requires_grad:
        raise ValueError("samples must require gradients")
    batch_size = logits_or_prediction.shape[0]
    if targets.ndim == 1 and targets.shape[0] == batch_size:
        scores = targets
    elif targets.ndim == 2 and targets.shape == (batch_size, 1):
        scores = targets[:, 0]
    else:
        raise ValueError("targets must contain one scalar score per sample")
    if not targets.is_floating_point():
        raise ValueError("discriminator scores must be floating-point tensors")
    if targets.device != logits_or_prediction.device:
        raise ValueError("samples and discriminator scores must be on the same device")
    if not targets.requires_grad:
        raise ValueError("discriminator scores must require gradients")

    penalties = []
    for index in range(batch_size):
        gradient = torch.autograd.grad(
            scores[index],
            logits_or_prediction,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if gradient is None:
            raise ValueError("discriminator scores must depend on the provided samples")
        penalties.append(gradient[index].reshape(-1).square().sum())

    result = torch.stack(penalties) * (gamma / 2.0)
    if reduction == "none":
        return result
    if reduction == "sum":
        return result.sum()
    return result.mean()
