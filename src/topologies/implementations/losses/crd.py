import math
import numbers

import torch
import torch.nn.functional as F


def crd(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if ignore_index is not None:
        raise ValueError("ignore_index must be None for CRD")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")

    negative_teacher_features = params.pop("negative_teacher_features", None)
    num_negatives = params.pop("num_negatives", 4096)
    dataset_size = params.pop("dataset_size", None)
    temperature = params.pop("temperature", 0.1)
    if params:
        raise ValueError(f"unsupported parameter(s): {', '.join(sorted(params))}")
    if negative_teacher_features is None:
        raise ValueError("negative_teacher_features is required")
    if isinstance(num_negatives, bool) or not isinstance(num_negatives, int) or num_negatives <= 0:
        raise ValueError("num_negatives must be a positive integer")
    if isinstance(dataset_size, bool) or not isinstance(dataset_size, int) or dataset_size <= 0:
        raise ValueError("dataset_size must be a positive integer")
    if isinstance(temperature, bool) or not isinstance(temperature, numbers.Real) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a finite positive real number")

    tensors = (logits_or_prediction, targets, negative_teacher_features)
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise ValueError("student, positive teacher, and negative teacher features must be tensors")
    if logits_or_prediction.ndim != 2 or targets.ndim != 2 or negative_teacher_features.ndim != 3:
        raise ValueError("student and positive teacher must have shape [B,D], and negatives must have shape [B,N,D]")
    batch_size, feature_size = logits_or_prediction.shape
    if batch_size <= 0 or feature_size <= 0 or targets.shape != logits_or_prediction.shape:
        raise ValueError("student and positive teacher tensors must have matching nonempty shapes [B,D]")
    if negative_teacher_features.shape[0] != batch_size or negative_teacher_features.shape[1] <= 0 or negative_teacher_features.shape[2] != feature_size:
        raise ValueError("negative teacher features must have matching shape [B,N,D] with nonempty axes")
    if negative_teacher_features.shape[1] != num_negatives:
        raise ValueError("num_negatives must match the negative teacher feature axis")
    if any(not tensor.is_floating_point() for tensor in tensors):
        raise ValueError("student, positive teacher, and negative teacher features must be floating point")
    if any(tensor.dtype != logits_or_prediction.dtype or tensor.device != logits_or_prediction.device for tensor in tensors[1:]):
        raise ValueError("student, positive teacher, and negative teacher features must share dtype and device")
    if any(not torch.isfinite(tensor).all().item() for tensor in tensors):
        raise ValueError("student, positive teacher, and negative teacher features must be finite")

    student_norm = torch.linalg.vector_norm(logits_or_prediction, ord=2, dim=-1, keepdim=True)
    positive_norm = torch.linalg.vector_norm(targets, ord=2, dim=-1, keepdim=True)
    negative_norm = torch.linalg.vector_norm(negative_teacher_features, ord=2, dim=-1, keepdim=True)
    if any((norm == 0).any().item() for norm in (student_norm, positive_norm, negative_norm)):
        raise ValueError("student, positive teacher, and negative teacher feature rows must have nonzero norm")
    if any(not torch.isfinite(norm).all().item() for norm in (student_norm, positive_norm, negative_norm)):
        raise ValueError("student, positive teacher, and negative teacher feature norms must be finite")

    student = logits_or_prediction / student_norm
    positive = targets / positive_norm
    negatives = negative_teacher_features / negative_norm
    log_correction = math.log(num_negatives) - math.log(dataset_size)
    positive_logits = (student * positive).sum(dim=-1) / temperature - log_correction
    negative_logits = (student.unsqueeze(1) * negatives).sum(dim=-1) / temperature - log_correction
    losses = F.softplus(-positive_logits) + num_negatives * F.softplus(negative_logits).mean(dim=1)

    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.mean()
