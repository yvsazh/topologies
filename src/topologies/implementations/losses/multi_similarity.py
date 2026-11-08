import math
import numbers

import torch


def multi_similarity(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if ignore_index is not None:
        raise ValueError("ignore_index must be None for embedding labels")
    if not isinstance(reduction, str) or reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    allowed_params = {"alpha", "beta", "lambda", "epsilon"}
    unexpected = set(params) - allowed_params
    if unexpected:
        raise ValueError(f"unsupported parameters: {', '.join(sorted(unexpected))}")

    defaults = {"alpha": 2.0, "beta": 50.0, "lambda": 1.0, "epsilon": 0.1}
    values = {}
    for name, default in defaults.items():
        value = params.get(name, default)
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite real scalar")
        values[name] = float(value)
    if values["alpha"] <= 0:
        raise ValueError("alpha must be positive")
    if values["beta"] <= 0:
        raise ValueError("beta must be positive")
    if values["epsilon"] < 0:
        raise ValueError("epsilon must be nonnegative")

    if not isinstance(logits_or_prediction, torch.Tensor) or not logits_or_prediction.is_floating_point():
        raise ValueError("logits_or_prediction must be a floating-point tensor")
    if logits_or_prediction.ndim != 2 or min(logits_or_prediction.shape) < 1:
        raise ValueError("logits_or_prediction must have nonempty shape [B, D]")
    if not bool(torch.isfinite(logits_or_prediction).all()):
        raise ValueError("logits_or_prediction must contain only finite values")
    if not isinstance(targets, torch.Tensor):
        exception_type = ValueError
        raise exception_type("targets must be a tensor of class IDs")
    if targets.ndim != 1 or targets.shape[0] != logits_or_prediction.shape[0]:
        raise ValueError("targets must have shape [B] matching the embedding batch")
    integer_dtypes = (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    )
    integer_dtypes += tuple(
        dtype for dtype in (getattr(torch, "uint16", None), getattr(torch, "uint32", None), getattr(torch, "uint64", None))
        if dtype is not None
    )
    if targets.dtype in integer_dtypes:
        pass
    elif targets.is_floating_point():
        if not bool(torch.isfinite(targets).all()) or not bool((targets == targets.round()).all()):
            raise ValueError("floating-point targets must contain finite integral class IDs")
    else:
        raise ValueError("targets must contain integral class IDs")
    if not bool(torch.isfinite(targets).all()):
        raise ValueError("targets must contain finite class IDs")
    if targets.device != logits_or_prediction.device:
        raise ValueError("targets and embeddings must be on the same device")

    alpha = values["alpha"]
    beta = values["beta"]
    margin = values["lambda"]
    epsilon = values["epsilon"]
    similarities = logits_or_prediction @ logits_or_prediction.transpose(0, 1)
    same_class = targets.unsqueeze(0) == targets.unsqueeze(1)
    diagonal = torch.eye(targets.shape[0], dtype=torch.bool, device=targets.device)
    positive_candidates = same_class & ~diagonal
    negative_candidates = ~same_class
    per_anchor = []

    for index in range(targets.shape[0]):
        positive_mask = positive_candidates[index]
        negative_mask = negative_candidates[index]
        row = similarities[index]
        if not bool(positive_mask.any()) or not bool(negative_mask.any()):
            per_anchor.append(row.sum() * 0.0)
            continue

        positive_values = row.masked_select(positive_mask)
        negative_values = row.masked_select(negative_mask)
        negative_mask = negative_mask & (row > positive_values.min() - epsilon)
        positive_mask = positive_mask & (row < negative_values.max() + epsilon)
        selected_positives = row.masked_select(positive_mask)
        selected_negatives = row.masked_select(negative_mask)
        positive_logsumexp = torch.logsumexp(
            torch.cat((row.new_zeros(1), -alpha * (selected_positives - margin))),
            dim=0,
        )
        negative_logsumexp = torch.logsumexp(
            torch.cat((row.new_zeros(1), beta * (selected_negatives - margin))),
            dim=0,
        )
        per_anchor.append(positive_logsumexp / alpha + negative_logsumexp / beta)

    losses = torch.stack(per_anchor)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.mean()
