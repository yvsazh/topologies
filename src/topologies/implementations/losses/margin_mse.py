import torch


def margin_mse(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if ignore_index is not None:
        raise ValueError("margin_mse does not support ignore_index")
    if params:
        raise ValueError(f"unsupported parameters: {sorted(params)}")
    if not isinstance(reduction, str) or reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if not torch.is_tensor(logits_or_prediction) or not torch.is_tensor(targets):
        raise ValueError("student scores and teacher targets must be tensors")
    if not logits_or_prediction.is_floating_point() or not targets.is_floating_point():
        raise ValueError("student scores and teacher targets must have floating-point dtypes")
    if (
        logits_or_prediction.ndim != 2
        or targets.ndim != 2
        or logits_or_prediction.shape[0] < 1
        or logits_or_prediction.shape[1] != 2
        or targets.shape != logits_or_prediction.shape
    ):
        raise ValueError("student scores and teacher targets must have matching shape [B, 2] with B >= 1")
    if logits_or_prediction.dtype != targets.dtype or logits_or_prediction.device != targets.device:
        raise ValueError("student scores and teacher targets must have matching dtype and device")
    if not torch.isfinite(logits_or_prediction).all() or not torch.isfinite(targets).all():
        raise ValueError("student scores and teacher targets must contain only finite values")

    student_margin = logits_or_prediction[:, 0] - logits_or_prediction[:, 1]
    teacher_scores = targets.detach()
    teacher_margin = teacher_scores[:, 0] - teacher_scores[:, 1]
    per_pair = (student_margin - teacher_margin).square()
    if reduction == "none":
        return per_pair
    if reduction == "sum":
        return per_pair.sum()
    return per_pair.mean()
