import math

import torch
import torch.nn.functional as F


def student_teacher(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if ignore_index is not None:
        raise ValueError("ignore_index must be None for teacher-logit targets")
    if type(reduction) is not str or reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    unsupported = set(params) - {"temperature"}
    if unsupported:
        raise ValueError(f"unsupported parameters: {', '.join(sorted(unsupported))}")
    temperature = params.get("temperature", 1.0)
    if type(temperature) not in (int, float):
        temperature = float("nan")
    else:
        try:
            temperature = float(temperature)
        except OverflowError:
            temperature = float("inf")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a finite positive real number")
    if not torch.is_tensor(logits_or_prediction) or not torch.is_tensor(targets):
        raise ValueError("student and teacher logits must be tensors")
    if logits_or_prediction.ndim != 2 or targets.ndim != 2:
        raise ValueError("student and teacher logits must have shape [B,C]")
    if logits_or_prediction.shape != targets.shape:
        raise ValueError("student and teacher logits must have matching shapes")
    if logits_or_prediction.shape[0] == 0 or logits_or_prediction.shape[1] < 2:
        raise ValueError("student and teacher logits must have shape [B,C] with B > 0 and C >= 2")
    if not torch.is_floating_point(logits_or_prediction) or not torch.is_floating_point(targets):
        raise ValueError("student and teacher logits must be floating point")
    if logits_or_prediction.dtype != targets.dtype or logits_or_prediction.device != targets.device:
        raise ValueError("student and teacher logits must share dtype and device")
    if not torch.isfinite(logits_or_prediction).all().item() or not torch.isfinite(targets).all().item():
        raise ValueError("student and teacher logits must be finite")
    student_log_probabilities = F.log_softmax(logits_or_prediction / temperature, dim=-1)
    teacher_log_probabilities = F.log_softmax(targets.detach() / temperature, dim=-1)
    losses = F.kl_div(
        student_log_probabilities,
        teacher_log_probabilities,
        reduction="none",
        log_target=True,
    ).sum(dim=-1) * temperature**2
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.mean()
