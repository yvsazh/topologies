import math
from numbers import Real

import torch
import torch.nn.functional as F


def dkd(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    allowed = {"teacher_logits", "alpha", "beta", "temperature"}
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(f"unexpected parameters: {sorted(unknown)}")
    missing = allowed - set(params)
    if missing:
        raise ValueError(f"missing required parameters: {sorted(missing)}")
    if not isinstance(logits_or_prediction, torch.Tensor):
        error = ValueError("logits_or_prediction must be a tensor")
        raise error
    if not logits_or_prediction.is_floating_point() or logits_or_prediction.ndim != 2:
        raise ValueError("logits_or_prediction must be a floating-point tensor with shape [B, C]")
    if logits_or_prediction.shape[0] == 0 or logits_or_prediction.shape[1] < 2:
        raise ValueError("logits_or_prediction must have nonempty batch and at least two classes")
    if not bool(torch.isfinite(logits_or_prediction).all()):
        raise ValueError("logits_or_prediction must contain only finite values")
    if not isinstance(targets, torch.Tensor) or targets.ndim != 1:
        raise ValueError("targets must be a one-dimensional tensor with shape [B]")
    if targets.shape[0] != logits_or_prediction.shape[0]:
        raise ValueError("targets must have shape [B] matching logits_or_prediction")
    if targets.device != logits_or_prediction.device:
        raise ValueError("targets and logits_or_prediction must be on the same device")
    if targets.dtype == torch.bool or targets.is_complex():
        raise ValueError("targets must contain class indices")
    if targets.is_floating_point() and (
        not bool(torch.isfinite(targets).all()) or not torch.equal(targets, targets.round())
    ):
        raise ValueError("floating-point targets must be finite integer class indices")
    if ignore_index is not None and (isinstance(ignore_index, bool) or not isinstance(ignore_index, int)):
        raise ValueError("ignore_index must be an integer or None")
    if not isinstance(reduction, str) or reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    for name in ("alpha", "beta", "temperature"):
        value = params[name]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite real number")
        if name == "temperature" and value <= 0:
            raise ValueError("temperature must be finite and positive")
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    temperature = float(params["temperature"])
    teacher_logits = params["teacher_logits"]
    if not isinstance(teacher_logits, torch.Tensor) or not teacher_logits.is_floating_point():
        raise ValueError("teacher_logits must be a floating-point tensor")
    if teacher_logits.shape != logits_or_prediction.shape:
        raise ValueError("teacher_logits must have the same shape as logits_or_prediction")
    if teacher_logits.device != logits_or_prediction.device:
        raise ValueError("teacher_logits and logits_or_prediction must be on the same device")
    if not bool(torch.isfinite(teacher_logits).all()):
        raise ValueError("teacher_logits must contain only finite values")
    target_indices = targets.long()
    valid = torch.ones_like(target_indices, dtype=torch.bool)
    if ignore_index is not None:
        valid = target_indices != ignore_index
    valid_targets = target_indices[valid]
    class_count = logits_or_prediction.shape[1]
    if valid_targets.numel() and bool(((valid_targets < 0) | (valid_targets >= class_count)).any()):
        raise ValueError("targets contain a class index outside logits_or_prediction")
    safe_targets = target_indices.masked_fill(~valid, 0)
    target_mask = F.one_hot(safe_targets, num_classes=class_count).to(dtype=torch.bool)
    student_logits = logits_or_prediction / temperature
    teacher_logits = teacher_logits.detach().to(dtype=logits_or_prediction.dtype) / temperature
    student_log_probabilities = F.log_softmax(student_logits, dim=-1)
    teacher_log_probabilities = F.log_softmax(teacher_logits, dim=-1)
    student_target_log_probability = student_log_probabilities.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
    teacher_target_log_probability = teacher_log_probabilities.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
    student_other_log_probability = torch.logsumexp(
        student_logits.masked_fill(target_mask, -torch.inf), dim=-1
    ) - torch.logsumexp(student_logits, dim=-1)
    teacher_other_log_probability = torch.logsumexp(
        teacher_logits.masked_fill(target_mask, -torch.inf), dim=-1
    ) - torch.logsumexp(teacher_logits, dim=-1)
    student_binary_log_probabilities = torch.stack(
        (student_target_log_probability, student_other_log_probability), dim=-1
    )
    teacher_binary_log_probabilities = torch.stack(
        (teacher_target_log_probability, teacher_other_log_probability), dim=-1
    )
    target_class_kl = (
        teacher_binary_log_probabilities.exp()
        * (teacher_binary_log_probabilities - student_binary_log_probabilities)
    ).sum(dim=-1)
    student_non_target_log_probabilities = F.log_softmax(
        student_logits.masked_fill(target_mask, -torch.inf), dim=-1
    ).masked_fill(target_mask, 0.0)
    teacher_non_target_log_probabilities = F.log_softmax(
        teacher_logits.masked_fill(target_mask, -torch.inf), dim=-1
    ).masked_fill(target_mask, 0.0)
    teacher_non_target_probabilities = teacher_non_target_log_probabilities.exp().masked_fill(target_mask, 0.0)
    non_target_class_kl = (
        teacher_non_target_probabilities
        * (teacher_non_target_log_probabilities - student_non_target_log_probabilities)
    ).sum(dim=-1)
    losses = (temperature**2 * (alpha * target_class_kl + beta * non_target_class_kl)).masked_fill(~valid, 0.0)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.sum() / valid.sum().clamp_min(1).to(dtype=losses.dtype)
