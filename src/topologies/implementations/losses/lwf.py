import math
from numbers import Real

import torch
import torch.nn.functional as F


def lwf(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    allowed = {"old_logits", "old_probabilities", "temperature", "lambda_old"}
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(f"unexpected parameters: {sorted(unknown)}")
    missing = {"old_logits", "old_probabilities"} - set(params)
    if missing:
        raise ValueError(f"missing required parameters: {sorted(missing)}")
    if not isinstance(logits_or_prediction, torch.Tensor):
        error = ValueError("logits_or_prediction must be a tensor")
        raise error
    if not logits_or_prediction.is_floating_point() or logits_or_prediction.ndim != 2:
        raise ValueError("logits_or_prediction must be a floating-point tensor with shape [B, C_new]")
    if logits_or_prediction.shape[0] == 0 or logits_or_prediction.shape[1] == 0:
        raise ValueError("logits_or_prediction must have nonempty batch and class dimensions")
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
    temperature = params.get("temperature", 2.0)
    lambda_old = params.get("lambda_old", 1.0)
    for name, value, minimum, inclusive in (
        ("temperature", temperature, 0.0, False),
        ("lambda_old", lambda_old, 0.0, True),
    ):
        if isinstance(value, bool) or not isinstance(value, Real):
            error = ValueError(f"{name} must be a finite real number")
            raise error
        if not math.isfinite(float(value)) or (value < minimum if inclusive else value <= minimum):
            qualifier = "nonnegative" if inclusive else "positive"
            raise ValueError(f"{name} must be finite and {qualifier}")
    temperature = float(temperature)
    lambda_old = float(lambda_old)
    old_logits = params["old_logits"]
    old_probabilities = params["old_probabilities"]
    if isinstance(old_logits, torch.Tensor):
        old_logit_tasks = (old_logits,)
    elif isinstance(old_logits, (tuple, list)) and old_logits:
        old_logit_tasks = tuple(old_logits)
    else:
        raise ValueError("old_logits must be a tensor or a nonempty sequence of tensors")
    if isinstance(old_probabilities, torch.Tensor):
        old_probability_tasks = (old_probabilities,)
    elif isinstance(old_probabilities, (tuple, list)) and old_probabilities:
        old_probability_tasks = tuple(old_probabilities)
    else:
        raise ValueError("old_probabilities must be a tensor or a nonempty sequence of tensors")
    if len(old_logit_tasks) != len(old_probability_tasks):
        raise ValueError("old_logits and old_probabilities must contain the same number of tasks")
    target_indices = targets.long()
    valid = torch.ones_like(target_indices, dtype=torch.bool)
    if ignore_index is not None:
        valid = target_indices != ignore_index
    valid_targets = target_indices[valid]
    if valid_targets.numel() and bool(((valid_targets < 0) | (valid_targets >= logits_or_prediction.shape[1])).any()):
        raise ValueError("targets contain a class index outside the new-task logits")
    new_losses = F.cross_entropy(
        logits_or_prediction,
        target_indices,
        reduction="none",
        ignore_index=-100 if ignore_index is None else ignore_index,
    )
    old_losses = torch.zeros_like(new_losses)
    for current_logits, teacher_probabilities in zip(old_logit_tasks, old_probability_tasks):
        if not isinstance(current_logits, torch.Tensor) or not current_logits.is_floating_point():
            raise ValueError("each old-task logit tensor must have a floating-point dtype")
        if current_logits.ndim != 2 or current_logits.shape[0] != logits_or_prediction.shape[0]:
            raise ValueError("each old-task logit tensor must have shape [B, C_old]")
        if current_logits.shape[1] == 0 or current_logits.device != logits_or_prediction.device:
            raise ValueError("each old-task logit tensor must have nonempty classes on the same device")
        if not isinstance(teacher_probabilities, torch.Tensor) or not teacher_probabilities.is_floating_point():
            raise ValueError("each old-task probability tensor must have a floating-point dtype")
        if teacher_probabilities.shape != current_logits.shape:
            raise ValueError("each old-task probability tensor must match its logit tensor shape")
        if teacher_probabilities.device != current_logits.device:
            raise ValueError("each old-task probability tensor must be on the same device as its logits")
        if not bool(torch.isfinite(teacher_probabilities).all()) or bool((teacher_probabilities < 0).any()):
            raise ValueError("old-task probabilities must be finite and nonnegative")
        row_sums = teacher_probabilities.sum(dim=-1)
        tolerance = max(1e-6, 10.0 * torch.finfo(teacher_probabilities.dtype).eps)
        if bool((row_sums <= 0).any()) or not torch.allclose(
            row_sums, torch.ones_like(row_sums), atol=tolerance, rtol=tolerance
        ):
            raise ValueError("each old-task probability row must sum to one")
        softened_teacher = teacher_probabilities.detach().to(dtype=current_logits.dtype).pow(1.0 / temperature)
        softened_teacher = softened_teacher / softened_teacher.sum(dim=-1, keepdim=True)
        current_log_probabilities = F.log_softmax(current_logits / temperature, dim=-1)
        old_losses = old_losses + -(softened_teacher * current_log_probabilities).sum(dim=-1).to(old_losses.dtype)
    losses = (new_losses + lambda_old * old_losses).masked_fill(~valid, 0.0)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.sum() / valid.sum().clamp_min(1).to(dtype=losses.dtype)
