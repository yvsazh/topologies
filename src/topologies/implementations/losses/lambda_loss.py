import math
from numbers import Real

import torch
import torch.nn.functional as functional


def lambda_loss(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    unknown = set(params) - {"sigma"}
    if unknown:
        raise ValueError(f"unexpected parameters: {sorted(unknown)}")
    if ignore_index is not None:
        raise ValueError("ignore_index must be None for LambdaLoss")
    if not isinstance(reduction, str) or reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    sigma = params.get("sigma", 1.0)
    if isinstance(sigma, bool) or not isinstance(sigma, Real):
        error = ValueError("sigma must be a finite positive real number")
        raise error
    try:
        sigma = float(sigma)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("sigma must be a finite positive real number") from error
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be a finite positive real number")
    if not isinstance(logits_or_prediction, torch.Tensor) or not logits_or_prediction.is_floating_point():
        raise ValueError("logits_or_prediction must be a floating-point tensor")
    if logits_or_prediction.ndim not in {1, 2} or logits_or_prediction.numel() == 0:
        raise ValueError("logits_or_prediction must have shape [N] or [B, N] with positive dimensions")
    if logits_or_prediction.shape[-1] == 0 or (logits_or_prediction.ndim == 2 and logits_or_prediction.shape[0] == 0):
        raise ValueError("logits_or_prediction must have shape [N] or [B, N] with positive dimensions")
    if not isinstance(targets, torch.Tensor) or targets.dtype == torch.bool or targets.is_complex():
        raise ValueError("targets must be a numeric tensor matching logits_or_prediction")
    if targets.shape != logits_or_prediction.shape:
        raise ValueError("targets must have the same shape as logits_or_prediction")
    if targets.device != logits_or_prediction.device:
        raise ValueError("targets and logits_or_prediction must be on the same device")
    if not bool(torch.isfinite(logits_or_prediction).all()):
        raise ValueError("logits_or_prediction must contain only finite scores")
    labels = targets.to(dtype=logits_or_prediction.dtype)
    if not bool(torch.isfinite(labels).all()) or bool((labels < 0).any()):
        raise ValueError("targets must contain finite, nonnegative relevance labels")
    scores = logits_or_prediction.unsqueeze(0) if logits_or_prediction.ndim == 1 else logits_or_prediction
    labels = labels.unsqueeze(0) if labels.ndim == 1 else labels
    per_query = []
    for query_scores, query_labels in zip(scores, labels):
        pair_indices = torch.triu_indices(
            query_labels.numel(), query_labels.numel(), offset=1, device=query_labels.device
        )
        left, right = pair_indices[0], pair_indices[1]
        preference = query_labels[left] != query_labels[right]
        left, right = left[preference], right[preference]
        if left.numel() == 0:
            per_query.append(query_scores[0] * 0)
            continue
        reverse = query_labels[left] < query_labels[right]
        high = torch.where(reverse, right, left)
        low = torch.where(reverse, left, right)
        score_order = torch.argsort(query_scores, descending=True, stable=True)
        ranks = torch.empty_like(score_order)
        ranks[score_order] = torch.arange(1, query_scores.numel() + 1, device=query_scores.device)
        score_discount = torch.reciprocal(torch.log2(ranks.to(query_scores.dtype) + 1))
        max_label = query_labels.max()
        scaled_gains = torch.exp2(query_labels - max_label) * (-torch.expm1(-query_labels * math.log(2)))
        ideal_order = torch.argsort(query_labels, descending=True, stable=True)
        ideal_discount = torch.reciprocal(
            torch.log2(torch.arange(1, query_labels.numel() + 1, dtype=query_scores.dtype, device=query_scores.device) + 1)
        )
        ideal_dcg = (scaled_gains[ideal_order] * ideal_discount).sum()
        normalized_gains = scaled_gains / ideal_dcg
        delta_ndcg = (normalized_gains[high] - normalized_gains[low]).abs()
        delta_ndcg = delta_ndcg * (score_discount[high] - score_discount[low]).abs()
        margins = sigma * (query_scores[high] - query_scores[low])
        pair_losses = delta_ndcg * functional.softplus(-margins) / math.log(2)
        per_query.append(pair_losses.sum())
    per_query_losses = torch.stack(per_query)
    if reduction == "none":
        return per_query_losses
    if reduction == "sum":
        return per_query_losses.sum()
    return per_query_losses.mean()
