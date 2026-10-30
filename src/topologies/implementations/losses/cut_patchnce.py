import math

import torch
import torch.nn.functional as functional


def cut_patchnce(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    temperature = params.pop("temperature", 0.07)
    if params:
        raise ValueError(f"unsupported parameters: {', '.join(sorted(params))}")
    if ignore_index is not None:
        raise ValueError("ignore_index must be None for continuous patch features")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if type(temperature) not in (int, float):
        raise ValueError("temperature must be finite and positive")
    try:
        temperature = float(temperature)
    except OverflowError:
        temperature = math.inf
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not isinstance(logits_or_prediction, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError("prediction and targets must be torch.Tensor instances")
    if logits_or_prediction.ndim != 3 or targets.ndim != 3:
        raise ValueError("prediction and targets must have rank 3 with shape [B, C, S]")
    if logits_or_prediction.shape != targets.shape:
        raise ValueError("prediction and targets must have identical shapes")
    if any(size == 0 for size in logits_or_prediction.shape):
        raise ValueError("prediction and targets must have nonempty batch, channel, and patch dimensions")
    if not logits_or_prediction.is_floating_point() or not targets.is_floating_point():
        raise ValueError("prediction and targets must use floating-point dtypes")
    if logits_or_prediction.dtype != targets.dtype:
        raise ValueError("prediction and targets must have the same dtype")
    if logits_or_prediction.device != targets.device:
        raise ValueError("prediction and targets must be on the same device")

    batch_size, _, patch_count = logits_or_prediction.shape
    query = functional.normalize(logits_or_prediction, p=2, dim=1)
    keys = functional.normalize(targets, p=2, dim=1)
    positive_logits = (query * keys).sum(dim=1, keepdim=False).unsqueeze(-1)
    negative_logits = torch.bmm(query.transpose(1, 2), keys)
    mask = torch.eye(patch_count, dtype=torch.bool, device=query.device).unsqueeze(0)
    negative_logits = negative_logits.masked_fill(mask, -torch.inf)
    logits = torch.cat((positive_logits, negative_logits), dim=-1).div(temperature)
    labels = torch.zeros(batch_size * patch_count, dtype=torch.long, device=query.device)
    per_patch = functional.cross_entropy(logits.reshape(-1, patch_count + 1), labels, reduction="none")
    if reduction == "none":
        return per_patch
    if reduction == "sum":
        return per_patch.sum()
    return per_patch.mean()
