import math
import numbers

import torch


def trades(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params) -> torch.Tensor:
    allowed_params = {"adversarial_logits", "beta"}
    unexpected = set(params) - allowed_params
    if unexpected:
        raise ValueError(f"unsupported parameters: {sorted(unexpected)}")
    if "adversarial_logits" not in params or "beta" not in params:
        raise ValueError("adversarial_logits and beta are required")
    adversarial_logits = params["adversarial_logits"]
    beta = params["beta"]
    if not isinstance(beta, numbers.Real) or isinstance(beta, bool) or not math.isfinite(float(beta)) or beta <= 0:
        raise ValueError("beta must be a finite positive real number")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if ignore_index is not None and (not isinstance(ignore_index, numbers.Integral) or isinstance(ignore_index, bool)):
        raise ValueError("ignore_index must be an integer or None")
    if not torch.is_tensor(logits_or_prediction) or not torch.is_tensor(adversarial_logits):
        raise ValueError("clean and adversarial logits must be tensors")
    if logits_or_prediction.ndim != 2 or adversarial_logits.ndim != 2:
        raise ValueError("clean and adversarial logits must have shape [B, C]")
    if logits_or_prediction.shape != adversarial_logits.shape:
        raise ValueError("clean and adversarial logits must have identical shapes")
    if logits_or_prediction.shape[0] == 0 or logits_or_prediction.shape[1] < 2:
        raise ValueError("logits must have a nonempty batch and at least two classes")
    if not torch.is_floating_point(logits_or_prediction) or not torch.is_floating_point(adversarial_logits):
        raise ValueError("clean and adversarial logits must be floating-point tensors")
    if logits_or_prediction.dtype != adversarial_logits.dtype or logits_or_prediction.device != adversarial_logits.device:
        raise ValueError("clean and adversarial logits must have matching dtype and device")
    if not torch.isfinite(logits_or_prediction).all() or not torch.isfinite(adversarial_logits).all():
        raise ValueError("clean and adversarial logits must contain only finite values")
    if not isinstance(targets, torch.Tensor) or targets.ndim != 1 or targets.shape[0] != logits_or_prediction.shape[0]:
        raise ValueError("targets must have shape [B] matching the logits batch")
    if targets.device != logits_or_prediction.device:
        raise ValueError("targets and logits must be on the same device")
    integer_dtypes = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
    if torch.is_floating_point(targets):
        if not torch.isfinite(targets).all() or not torch.equal(targets, targets.round()):
            raise ValueError("floating-point targets must contain finite integer class indices")
    elif targets.dtype not in integer_dtypes:
        raise ValueError("targets must contain integer class indices")
    labels = targets.long()
    if ignore_index is None:
        valid = torch.ones_like(labels, dtype=torch.bool)
    else:
        valid = labels != int(ignore_index)
    invalid_labels = valid & ((labels < 0) | (labels >= logits_or_prediction.shape[1]))
    if invalid_labels.any():
        raise ValueError("non-ignored targets must be valid class indices")
    safe_labels = labels.masked_fill(~valid, 0)
    clean_log_probabilities = torch.log_softmax(logits_or_prediction, dim=-1)
    adversarial_probabilities = torch.softmax(adversarial_logits, dim=-1)
    natural_loss = -clean_log_probabilities.gather(1, safe_labels.unsqueeze(1)).squeeze(1)
    pairwise_loss = -(adversarial_probabilities * clean_log_probabilities).sum(dim=-1)
    per_example = (natural_loss + float(beta) * pairwise_loss).masked_fill(~valid, 0)
    if reduction == "none":
        return per_example
    if reduction == "sum":
        return per_example.sum()
    denominator = valid.sum().clamp_min(1).to(dtype=per_example.dtype)
    return per_example.sum() / denominator
