import math
import numbers

import torch
import torch.nn.functional as F


def ge2e(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if ignore_index is not None:
        raise ValueError("ge2e does not support ignore_index.")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'.")
    if not torch.is_tensor(logits_or_prediction):
        raise ValueError("logits_or_prediction must be a torch.Tensor.")
    if (
        logits_or_prediction.ndim != 2
        or logits_or_prediction.shape[0] == 0
        or logits_or_prediction.shape[1] == 0
    ):
        raise ValueError("logits_or_prediction must have shape [B, D] with positive dimensions.")
    if not logits_or_prediction.is_floating_point():
        raise ValueError("logits_or_prediction must have a floating-point dtype.")
    if not torch.is_tensor(targets):
        raise ValueError("targets must be an integer torch.Tensor of speaker IDs.")
    if targets.ndim != 1 or targets.shape[0] != logits_or_prediction.shape[0]:
        raise ValueError("targets must have shape [B] matching logits_or_prediction.")
    if targets.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise ValueError("targets must have an integer dtype.")
    if targets.device != logits_or_prediction.device:
        raise ValueError("targets and logits_or_prediction must be on the same device.")

    objective = params.pop("objective", "softmax")
    scale = params.pop("w", 10.0)
    bias = params.pop("b", -5.0)
    if params:
        names = ", ".join(sorted(params))
        raise ValueError(f"unsupported ge2e parameter(s): {names}.")
    if objective not in ("softmax", "contrast"):
        raise ValueError("objective must be 'softmax' or 'contrast'.")
    for name, value, positive in (("w", scale, True), ("b", bias, False)):
        if torch.is_tensor(value):
            if value.numel() != 1 or not value.is_floating_point():
                raise ValueError(f"{name} must be a floating-point scalar.")
            if value.device != logits_or_prediction.device:
                raise ValueError(f"{name} must be on the same device as logits_or_prediction.")
            numeric_value = value.detach().item()
        elif isinstance(value, numbers.Real) and not isinstance(value, bool):
            numeric_value = float(value)
        else:
            raise ValueError(
                f"{name} must be a finite real scalar or floating-point tensor scalar."
            )
        if not math.isfinite(numeric_value) or (positive and numeric_value <= 0):
            requirement = "positive and finite" if positive else "finite"
            raise ValueError(f"{name} must be {requirement}.")

    _, speaker_indices, counts = torch.unique(
        targets, sorted=True, return_inverse=True, return_counts=True
    )
    if counts.numel() < 2:
        raise ValueError("targets must contain at least two distinct speakers.")
    if counts[0].item() < 2 or not torch.equal(counts, counts[0].expand_as(counts)):
        raise ValueError("targets must contain the same number of at least two utterances per speaker.")

    speaker_count = counts.numel()
    utterance_count = int(counts[0].item())
    sort_indices = torch.argsort(speaker_indices)
    grouped = logits_or_prediction.index_select(0, sort_indices).reshape(
        speaker_count, utterance_count, logits_or_prediction.shape[1]
    )
    embeddings = F.normalize(grouped, p=2, dim=-1)
    centroids = embeddings.mean(dim=1)
    leave_one_out = (embeddings.sum(dim=1, keepdim=True) - embeddings) / (utterance_count - 1)
    expanded_embeddings = embeddings.unsqueeze(2).expand(
        speaker_count, utterance_count, speaker_count, embeddings.shape[-1]
    )
    all_similarities = F.cosine_similarity(
        expanded_embeddings, centroids.view(1, 1, speaker_count, -1), dim=-1
    )
    positive_similarities = F.cosine_similarity(embeddings, leave_one_out, dim=-1)
    positive_mask = torch.eye(speaker_count, dtype=torch.bool, device=embeddings.device).unsqueeze(1)
    cosine_similarities = torch.where(
        positive_mask, positive_similarities.unsqueeze(-1), all_similarities
    )
    similarities = scale * cosine_similarities + bias

    speaker_labels = torch.arange(speaker_count, device=embeddings.device).view(-1, 1)
    if objective == "softmax":
        losses = F.cross_entropy(
            similarities.reshape(speaker_count * utterance_count, speaker_count),
            speaker_labels.expand(speaker_count, utterance_count).reshape(-1),
            reduction="none",
        )
    else:
        labels = speaker_labels.view(speaker_count, 1, 1).expand(
            speaker_count, utterance_count, 1
        )
        positive_scores = similarities.gather(-1, labels).squeeze(-1)
        negative_scores = similarities.masked_fill(positive_mask, -torch.inf).max(dim=-1).values
        losses = (1.0 - torch.sigmoid(positive_scores) + torch.sigmoid(negative_scores)).reshape(-1)

    losses = losses.index_select(0, torch.argsort(sort_indices))
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.mean()
