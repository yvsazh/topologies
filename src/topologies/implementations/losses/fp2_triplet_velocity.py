import math
import numbers

import torch


def fp2_triplet_velocity(
    logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params
):
    if ignore_index is not None:
        raise ValueError("fp2_triplet_velocity does not support ignore_index.")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'.")
    if not torch.is_tensor(logits_or_prediction):
        raise ValueError("logits_or_prediction must be a torch.Tensor.")
    if not torch.is_tensor(targets):
        raise ValueError("targets must be a torch.Tensor.")
    if (
        logits_or_prediction.ndim < 2
        or logits_or_prediction.shape[0] == 0
        or any(size == 0 for size in logits_or_prediction.shape[1:])
    ):
        raise ValueError("logits_or_prediction must have shape [B, ...] with positive dimensions.")
    if not logits_or_prediction.is_floating_point():
        raise ValueError("logits_or_prediction must have a floating-point dtype.")
    if (
        targets.shape != logits_or_prediction.shape
        or targets.dtype != logits_or_prediction.dtype
        or targets.device != logits_or_prediction.device
        or not targets.is_floating_point()
    ):
        raise ValueError("targets must match logits_or_prediction shape, dtype, and device.")

    preferences = params.pop("preferences", None)
    margin = params.pop("margin", 1.0)
    if params:
        names = ", ".join(sorted(params))
        raise ValueError(f"unsupported fp2_triplet_velocity parameter(s): {names}.")
    if not torch.is_tensor(preferences):
        raise ValueError("preferences must be a torch.Tensor with shape [B, P].")
    if (
        preferences.ndim != 2
        or preferences.shape[0] != logits_or_prediction.shape[0]
        or preferences.shape[1] == 0
        or not preferences.is_floating_point()
        or preferences.device != logits_or_prediction.device
    ):
        raise ValueError("preferences must be a floating-point tensor [B, P] on the prediction device.")
    if not bool(torch.isfinite(preferences).all()):
        raise ValueError("preferences must contain only finite values.")
    if (
        isinstance(margin, bool)
        or not isinstance(margin, numbers.Real)
        or not math.isfinite(margin)
        or margin <= 0
    ):
        raise ValueError("margin must be a positive finite scalar.")

    batch_size = logits_or_prediction.shape[0]
    prediction_flat = logits_or_prediction.reshape(batch_size, -1)
    targets_flat = targets.reshape(batch_size, -1)
    if batch_size == 1:
        losses = (prediction_flat.sum(dim=-1) + targets_flat.sum(dim=-1)) * 0
    else:
        preference_distances = (
            preferences[:, None, :] - preferences[None, :, :]
        ).square().sum(dim=-1)
        diagonal = torch.eye(batch_size, dtype=torch.bool, device=preferences.device)
        neighbor_indices = preference_distances.masked_fill(diagonal, float("inf")).argmin(dim=1)
        positive = (prediction_flat - targets_flat).square().mean(dim=-1)
        negative = (prediction_flat - prediction_flat[neighbor_indices]).square().mean(dim=-1)
        losses = (positive - margin * negative).clamp_min(0)

    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.mean()
