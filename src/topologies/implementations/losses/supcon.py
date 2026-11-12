import math

import torch


def supcon(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    temperature = params.pop("temperature", 0.1)
    if params:
        raise ValueError(f"unexpected parameters: {', '.join(sorted(params))}")
    if ignore_index is not None:
        raise ValueError("ignore_index is not supported")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    temperature_valid = (
        isinstance(temperature, (int, float))
        and not isinstance(temperature, bool)
        and math.isfinite(temperature)
        and temperature > 0
    )
    if not temperature_valid:
        raise ValueError("temperature must be a finite positive scalar")
    if not isinstance(logits_or_prediction, torch.Tensor) or not logits_or_prediction.is_floating_point():
        raise ValueError("logits_or_prediction must be a floating-point tensor")
    if logits_or_prediction.ndim != 3:
        raise ValueError("logits_or_prediction must have shape [B, V, D]")
    batch_size, view_count, embedding_size = logits_or_prediction.shape
    if batch_size < 1 or view_count < 2 or embedding_size < 1:
        raise ValueError("logits_or_prediction must have B >= 1, V >= 2, and D >= 1")
    if not isinstance(targets, torch.Tensor) or targets.ndim != 1 or targets.shape[0] != batch_size:
        raise ValueError("targets must have shape [B]")
    if targets.is_complex() or targets.dtype == torch.bool:
        raise ValueError("targets must contain integer class labels")
    if targets.is_floating_point() and (
        not torch.isfinite(targets).all() or not torch.equal(targets, targets.round())
    ):
        raise ValueError("targets must contain integer class labels")
    labels = targets.to(dtype=torch.long).repeat_interleave(view_count)
    if targets.device != logits_or_prediction.device:
        raise ValueError("targets and logits_or_prediction must be on the same device")
    if not torch.isfinite(logits_or_prediction).all():
        raise ValueError("logits_or_prediction must contain only finite values")

    norms = torch.linalg.vector_norm(logits_or_prediction, ord=2, dim=-1, keepdim=True)
    if (norms == 0).any():
        raise ValueError("embedding vectors must have nonzero norm")
    embeddings = (logits_or_prediction / norms).reshape(batch_size * view_count, embedding_size)
    similarities = embeddings @ embeddings.transpose(0, 1) / temperature
    self_mask = torch.eye(similarities.shape[0], dtype=torch.bool, device=similarities.device)
    positive_mask = (labels[:, None] == labels[None, :]) & ~self_mask
    logits = similarities.masked_fill(self_mask, -torch.inf)
    log_probabilities = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_log_probabilities = torch.where(
        positive_mask, log_probabilities, torch.zeros_like(log_probabilities)
    )
    anchor_losses = -positive_log_probabilities.sum(dim=1) / positive_mask.sum(dim=1)

    if reduction == "none":
        return anchor_losses
    if reduction == "sum":
        return anchor_losses.sum()
    return anchor_losses.mean()
