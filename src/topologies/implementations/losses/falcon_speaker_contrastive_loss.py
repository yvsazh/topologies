import math
import numbers

import torch
import torch.nn.functional as F


def falcon_speaker_contrastive_loss(
    logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params
):
    if ignore_index is not None:
        raise ValueError("ignore_index must be None because this loss defines no ignored samples")
    if not isinstance(reduction, str) or reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if not isinstance(logits_or_prediction, torch.Tensor) or not logits_or_prediction.is_floating_point():
        raise ValueError("logits_or_prediction must be a floating-point tensor")
    if not isinstance(targets, torch.Tensor) or not targets.is_floating_point():
        raise ValueError("targets must be a floating-point tensor")
    if logits_or_prediction.ndim != 2 or min(logits_or_prediction.shape) < 1:
        raise ValueError("logits_or_prediction must be a non-empty [B,D] tensor")
    if targets.shape != logits_or_prediction.shape:
        raise ValueError("targets must have the same [B,D] shape as logits_or_prediction")
    if targets.dtype != logits_or_prediction.dtype:
        raise ValueError("targets and logits_or_prediction must have the same dtype")
    if targets.device != logits_or_prediction.device:
        raise ValueError("targets and logits_or_prediction must be on the same device")

    negatives = params.pop("negatives", None)
    negative_mask = params.pop("negative_mask", None)
    margin = params.pop("margin", 0.5)
    tau = params.pop("tau", 1.0)
    if params:
        raise ValueError("unsupported parameters")
    for name, value in (("margin", margin), ("tau", tau)):
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite real number")
    margin = float(margin)
    tau = float(tau)

    batch_size, embedding_dim = logits_or_prediction.shape
    if negatives is None:
        negatives = logits_or_prediction.new_empty((batch_size, 0, embedding_dim))
    elif not isinstance(negatives, torch.Tensor) or not negatives.is_floating_point():
        raise ValueError("negatives must be a floating-point tensor")
    if negatives.ndim != 3 or negatives.shape[0] != batch_size or negatives.shape[2] != embedding_dim:
        raise ValueError("negatives must have shape [B,K,D] matching logits_or_prediction")
    if negatives.dtype != logits_or_prediction.dtype:
        raise ValueError("negatives and logits_or_prediction must have the same dtype")
    if negatives.device != logits_or_prediction.device:
        raise ValueError("negatives and logits_or_prediction must be on the same device")

    if negative_mask is None:
        negative_mask = torch.ones(
            negatives.shape[:2], dtype=torch.bool, device=logits_or_prediction.device
        )
    else:
        try:
            negative_mask = torch.as_tensor(
                negative_mask, dtype=torch.bool, device=logits_or_prediction.device
            )
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError("negative_mask must be boolean-convertible") from error
        if negative_mask.shape != negatives.shape[:2]:
            raise ValueError("negative_mask must have shape [B,K] matching negatives")

    query = logits_or_prediction
    positive = targets.detach()
    negatives = negatives.detach()
    positive_similarities = F.cosine_similarity(query, positive, dim=-1)
    losses = []
    for index in range(batch_size):
        loss = 1.0 - positive_similarities[index]
        valid_negatives = negatives[index][negative_mask[index]]
        if valid_negatives.shape[0] > 0:
            repeated_query = query[index].unsqueeze(0).expand_as(valid_negatives)
            negative_similarities = F.cosine_similarity(repeated_query, valid_negatives, dim=-1)
            weights = torch.softmax(tau * negative_similarities, dim=0)
            loss = loss + (weights * F.relu(negative_similarities - margin)).sum()
        losses.append(loss)

    per_sample = torch.stack(losses)
    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    return per_sample.mean()
