import math
import numbers

import torch


def dist(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if ignore_index is not None:
        raise ValueError("ignore_index must be None because DIST defines no ignored entries")
    if not isinstance(logits_or_prediction, torch.Tensor) or not logits_or_prediction.is_floating_point():
        raise ValueError("logits_or_prediction must be a floating-point tensor")
    if not isinstance(targets, torch.Tensor) or not targets.is_floating_point():
        raise ValueError("targets must be a floating-point tensor")
    if logits_or_prediction.ndim != 2 or min(logits_or_prediction.shape) < 1:
        raise ValueError("logits_or_prediction must be a non-empty [B,C] probability tensor")
    if targets.ndim != 2 or min(targets.shape) < 1:
        raise ValueError("targets must be a non-empty [B,C] probability tensor")
    if logits_or_prediction.shape != targets.shape:
        raise ValueError("logits_or_prediction and targets must have identical [B,C] shapes")
    if logits_or_prediction.dtype != targets.dtype:
        raise ValueError("logits_or_prediction and targets must have matching dtypes")
    if logits_or_prediction.device != targets.device:
        raise ValueError("logits_or_prediction and targets must be on the same device")
    if not isinstance(reduction, str) or reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")

    beta = params.pop("beta", 1.0)
    gamma = params.pop("gamma", 1.0)
    if params:
        raise ValueError("unsupported parameters")
    if isinstance(beta, bool) or not isinstance(beta, numbers.Real) or not math.isfinite(float(beta)) or beta < 0:
        raise ValueError("beta must be a finite nonnegative real number")
    if isinstance(gamma, bool) or not isinstance(gamma, numbers.Real) or not math.isfinite(float(gamma)) or gamma < 0:
        raise ValueError("gamma must be a finite nonnegative real number")
    beta = float(beta)
    gamma = float(gamma)

    student = logits_or_prediction
    teacher = targets.detach()
    if not bool(torch.isfinite(student).all()) or not bool(torch.isfinite(teacher).all()):
        raise ValueError("student and teacher probabilities must be finite")
    if bool((student < 0).any()) or bool((student > 1).any()):
        raise ValueError("logits_or_prediction must contain probabilities in [0,1]")
    if bool((teacher < 0).any()) or bool((teacher > 1).any()):
        raise ValueError("targets must contain probabilities in [0,1]")
    tolerance = max(1e-6, float(torch.finfo(student.dtype).eps) * 4.0)
    expected_sums = torch.ones(student.shape[0], dtype=student.dtype, device=student.device)
    if not torch.allclose(student.sum(dim=1), expected_sums, atol=tolerance, rtol=tolerance):
        raise ValueError("each logits_or_prediction row must sum to one")
    if not torch.allclose(teacher.sum(dim=1), expected_sums, atol=tolerance, rtol=tolerance):
        raise ValueError("each target row must sum to one")

    def pearson_distance(left, right, dim):
        left_centered = left - left.mean(dim=dim, keepdim=True)
        right_centered = right - right.mean(dim=dim, keepdim=True)
        left_squared_norm = left_centered.square().sum(dim=dim)
        right_squared_norm = right_centered.square().sum(dim=dim)
        if bool((left_squared_norm <= 0).any()) or bool((right_squared_norm <= 0).any()):
            raise ValueError("student and teacher probability vectors must have non-zero variance")
        correlation = (left_centered * right_centered).sum(dim=dim) / torch.sqrt(
            left_squared_norm * right_squared_norm
        )
        return 1.0 - correlation

    inter = pearson_distance(student, teacher, dim=1)
    intra = pearson_distance(student, teacher, dim=0)
    per_example = beta * inter + gamma * intra.mean().expand_as(inter)
    if reduction == "none":
        return per_example
    if reduction == "sum":
        return per_example.sum()
    return per_example.mean()
