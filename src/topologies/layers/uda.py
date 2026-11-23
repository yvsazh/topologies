import math

import torch
from torch import nn
from torch.nn import functional as F


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


class UDA(nn.Module):
    component_id = "uda"

    def __init__(self, unsupervised_weight=1.0, temperature=1.0, confidence_threshold=None,
                 reduction="mean"):
        super().__init__()
        if (
            not _finite_number(unsupervised_weight)
            or unsupervised_weight < 0
        ):
            raise ValueError("unsupervised_weight must be a finite nonnegative number")
        if (
            not _finite_number(temperature)
            or temperature <= 0
        ):
            raise ValueError("temperature must be a finite positive number")
        if confidence_threshold is not None and (
            not _finite_number(confidence_threshold)
            or not 0 <= confidence_threshold <= 1
        ):
            raise ValueError("confidence_threshold must be None or a finite number in [0, 1]")
        if reduction not in ("mean", "none"):
            raise ValueError("reduction must be 'mean' or 'none'")
        self.unsupervised_weight = float(unsupervised_weight)
        self.temperature = float(temperature)
        self.confidence_threshold = (
            None if confidence_threshold is None else float(confidence_threshold)
        )
        self.reduction = reduction

    def forward(self, labeled_logits, labels, teacher_logits, student_logits):
        tensors = {
            "labeled_logits": labeled_logits,
            "labels": labels,
            "teacher_logits": teacher_logits,
            "student_logits": student_logits,
        }
        if any(not isinstance(value, torch.Tensor) for value in tensors.values()):
            raise ValueError("labeled_logits, labels, teacher_logits, and student_logits must be tensors")
        if labeled_logits.ndim != 2 or teacher_logits.ndim != 2 or student_logits.ndim != 2:
            raise ValueError("logits must be 2D tensors shaped [batch, classes]")
        if labels.ndim == 2 and labels.shape == (labeled_logits.shape[0], 1):
            labels = labels.squeeze(-1)
        if labels.ndim != 1 or labels.shape[0] != labeled_logits.shape[0]:
            raise ValueError("labels must have shape [labeled_batch]")
        if labeled_logits.shape[0] < 1 or teacher_logits.shape[0] < 1:
            raise ValueError("labeled and unlabeled batch dimensions must be positive")
        if labeled_logits.shape[1] < 2:
            raise ValueError("logits must contain at least two classes")
        if labeled_logits.shape[1] != teacher_logits.shape[1] or teacher_logits.shape != student_logits.shape:
            raise ValueError("all logits must have the same class count and teacher/student shapes must match")
        if self.reduction == "none" and labeled_logits.shape[0] != teacher_logits.shape[0]:
            raise ValueError("reduction='none' requires equal labeled and unlabeled batch sizes")
        if any(
            not logits.is_floating_point()
            for logits in (labeled_logits, teacher_logits, student_logits)
        ):
            raise ValueError("logits must have floating-point values")
        if labels.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            if not labels.is_floating_point():
                raise ValueError("labels must contain integer class indices")
            if not bool(torch.isfinite(labels).all()) or not bool(torch.equal(labels, labels.round())):
                raise ValueError("labels must contain finite integer class indices")
        if labels.device != labeled_logits.device or any(
            logits.device != labeled_logits.device for logits in (teacher_logits, student_logits)
        ):
            raise ValueError("labels and all logits must be on the same device")
        labels = labels.long()
        if bool((labels < 0).any()) or bool((labels >= labeled_logits.shape[1]).any()):
            raise ValueError("labels contain a class index outside the logits class range")

        supervised_loss = F.cross_entropy(labeled_logits, labels, reduction=self.reduction)
        detached_teacher_logits = teacher_logits.detach()
        teacher_probabilities = torch.softmax(detached_teacher_logits, dim=-1)
        sharpened_targets = torch.softmax(detached_teacher_logits / self.temperature, dim=-1)
        student_log_probabilities = torch.log_softmax(student_logits, dim=-1)
        per_example_loss = -(sharpened_targets * student_log_probabilities).sum(dim=-1)
        if self.confidence_threshold is not None:
            mask = teacher_probabilities.max(dim=-1).values > self.confidence_threshold
            per_example_loss = per_example_loss * mask.to(per_example_loss.dtype)
        unsupervised_loss = (
            per_example_loss.mean() if self.reduction == "mean" else per_example_loss
        )
        return supervised_loss + self.unsupervised_weight * unsupervised_loss
