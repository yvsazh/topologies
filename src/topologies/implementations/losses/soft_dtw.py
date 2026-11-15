import math
import numbers

import torch


def soft_dtw(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if ignore_index is not None:
        raise ValueError("ignore_index is unsupported for continuous sequence targets")
    if not isinstance(reduction, str) or reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    unknown = set(params) - {"gamma"}
    if unknown:
        raise ValueError(f"unexpected parameters: {sorted(unknown)}")

    gamma = params.get("gamma", 1.0)
    if (
        isinstance(gamma, bool)
        or not isinstance(gamma, numbers.Real)
        or not math.isfinite(float(gamma))
        or gamma <= 0.0
    ):
        raise ValueError("gamma must be a finite positive real scalar")
    gamma = float(gamma)

    if (
        not isinstance(logits_or_prediction, torch.Tensor)
        or not isinstance(targets, torch.Tensor)
        or not logits_or_prediction.is_floating_point()
        or not targets.is_floating_point()
    ):
        raise ValueError("prediction and targets must be floating-point tensors")
    if logits_or_prediction.ndim not in {2, 3} or targets.ndim != logits_or_prediction.ndim:
        raise ValueError("prediction and targets must have shapes [B, N] or [B, N, D]")
    if logits_or_prediction.shape[0] != targets.shape[0]:
        raise ValueError("prediction and targets must have matching batch sizes for [B, N] shapes")
    if logits_or_prediction.ndim == 3 and logits_or_prediction.shape[2] != targets.shape[2]:
        raise ValueError("prediction and targets must have matching feature dimensions")
    if logits_or_prediction.shape[0] < 1 or logits_or_prediction.shape[1] < 1 or targets.shape[1] < 1:
        raise ValueError("prediction and targets must contain nonempty batches and sequences")
    if logits_or_prediction.ndim == 3 and logits_or_prediction.shape[2] < 1:
        raise ValueError("prediction and targets must have nonempty feature dimensions")
    if (
        logits_or_prediction.dtype != targets.dtype
        or logits_or_prediction.device != targets.device
    ):
        raise ValueError("prediction and targets must have matching dtype and device")
    if not torch.isfinite(logits_or_prediction).all() or not torch.isfinite(targets).all():
        raise ValueError("prediction and targets must contain only finite values")

    if logits_or_prediction.ndim == 2:
        logits_or_prediction = logits_or_prediction.unsqueeze(-1)
        targets = targets.unsqueeze(-1)
    costs = (logits_or_prediction.unsqueeze(2) - targets.unsqueeze(1)).square().sum(dim=-1)
    batch_size, prediction_length, target_length = costs.shape
    boundary = torch.full(
        (batch_size,), float("inf"), dtype=costs.dtype, device=costs.device
    )
    table = [[torch.zeros_like(boundary)] + [boundary] * target_length]

    for i in range(1, prediction_length + 1):
        row = [boundary]
        for j in range(1, target_length + 1):
            predecessors = torch.stack(
                (table[i - 1][j - 1], table[i - 1][j], row[j - 1]), dim=0
            )
            soft_minimum = -gamma * torch.logsumexp(-predecessors / gamma, dim=0)
            row.append(costs[:, i - 1, j - 1] + soft_minimum)
        table.append(row)

    per_sample = table[prediction_length][target_length]
    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    return per_sample.mean()
