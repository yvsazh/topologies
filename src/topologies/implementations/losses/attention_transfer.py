import math
import numbers

import torch


def attention_transfer(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if ignore_index is not None:
        raise ValueError("ignore_index must be None for activation targets")
    if not isinstance(logits_or_prediction, torch.Tensor) or not logits_or_prediction.is_floating_point():
        raise ValueError("logits_or_prediction must be a floating-point tensor")
    if not isinstance(targets, torch.Tensor) or not targets.is_floating_point():
        raise ValueError("targets must be a floating-point tensor")
    if logits_or_prediction.ndim != 4 or min(logits_or_prediction.shape) < 1:
        raise ValueError("logits_or_prediction must be a non-empty [B,C,H,W] tensor")
    if targets.ndim != 4 or min(targets.shape) < 1:
        raise ValueError("targets must be a non-empty [B,C,H,W] tensor")
    if logits_or_prediction.shape[0] != targets.shape[0]:
        raise ValueError("logits_or_prediction and targets must have matching batch dimensions")
    if logits_or_prediction.shape[-2:] != targets.shape[-2:]:
        raise ValueError("logits_or_prediction and targets must have matching spatial shapes")
    if logits_or_prediction.dtype != targets.dtype:
        raise ValueError("logits_or_prediction and targets must have matching dtypes")
    if logits_or_prediction.device != targets.device:
        raise ValueError("logits_or_prediction and targets must be on the same device")
    if not isinstance(reduction, str) or reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")

    beta = params.pop("beta", None)
    if params:
        raise ValueError("unsupported parameters")
    if beta is None:
        beta = 1000.0 / (logits_or_prediction.shape[0] * logits_or_prediction.shape[2] * logits_or_prediction.shape[3])
    elif isinstance(beta, bool) or not isinstance(beta, numbers.Real) or not math.isfinite(float(beta)) or beta < 0:
        raise ValueError("beta must be a finite nonnegative real number or None")
    beta = float(beta)

    student_maps = logits_or_prediction.square().sum(dim=1).flatten(start_dim=1)
    teacher_maps = targets.detach().square().sum(dim=1).flatten(start_dim=1)
    student_norms = torch.linalg.vector_norm(student_maps, ord=2, dim=1, keepdim=True)
    teacher_norms = torch.linalg.vector_norm(teacher_maps, ord=2, dim=1, keepdim=True)
    if bool((student_norms == 0).any()) or bool((teacher_norms == 0).any()):
        raise ValueError("student and teacher attention maps must have non-zero norms; zero-norm maps are invalid")

    student_maps = student_maps / student_norms
    teacher_maps = teacher_maps / teacher_norms
    losses = (beta / 2.0) * torch.linalg.vector_norm(student_maps - teacher_maps, ord=2, dim=1)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.mean()
