import torch
import torch.nn.functional as F


def group_dro(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    unknown = set(params) - {"group_ids"}
    if unknown:
        raise ValueError(f"unexpected parameters: {sorted(unknown)}")
    if "group_ids" not in params:
        raise ValueError("missing required parameter: group_ids")
    group_ids = params["group_ids"]
    if not isinstance(logits_or_prediction, torch.Tensor) or not logits_or_prediction.is_floating_point():
        raise ValueError("logits_or_prediction must be a floating-point tensor")
    if logits_or_prediction.ndim != 2:
        raise ValueError("logits_or_prediction must have shape [B, C]")
    if logits_or_prediction.shape[0] == 0 or logits_or_prediction.shape[1] < 2:
        raise ValueError("logits_or_prediction must have a nonempty batch and at least two classes")
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
    if not isinstance(group_ids, torch.Tensor) or group_ids.ndim != 1:
        raise ValueError("group_ids must be a one-dimensional tensor with shape [B]")
    if group_ids.shape[0] != logits_or_prediction.shape[0]:
        raise ValueError("group_ids must have shape [B] matching logits_or_prediction")
    if group_ids.device != logits_or_prediction.device:
        raise ValueError("group_ids and logits_or_prediction must be on the same device")
    if group_ids.dtype == torch.bool or group_ids.is_complex():
        raise ValueError("group_ids must contain integer-valued group IDs")
    if group_ids.is_floating_point() and (
        not bool(torch.isfinite(group_ids).all()) or not torch.equal(group_ids, group_ids.round())
    ):
        raise ValueError("group_ids must contain finite integer-valued group IDs")
    if ignore_index is not None and (isinstance(ignore_index, bool) or not isinstance(ignore_index, int)):
        raise ValueError("ignore_index must be an integer or None")
    if not isinstance(reduction, str) or reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    target_indices = targets.long()
    valid = torch.ones_like(target_indices, dtype=torch.bool)
    if ignore_index is not None:
        valid = target_indices != ignore_index
    valid_targets = target_indices[valid]
    class_count = logits_or_prediction.shape[1]
    if valid_targets.numel() and bool(((valid_targets < 0) | (valid_targets >= class_count)).any()):
        raise ValueError("targets contain a class index outside logits_or_prediction")
    if not bool(valid.any()):
        raise ValueError("at least one non-ignored example is required")
    valid_groups = group_ids.long()[valid]
    per_example_losses = F.cross_entropy(logits_or_prediction[valid], valid_targets, reduction="none")
    _, group_index = torch.unique(valid_groups, sorted=True, return_inverse=True)
    group_count = int(group_index.max().item()) + 1
    group_sums = per_example_losses.new_zeros(group_count).scatter_add(0, group_index, per_example_losses)
    group_sizes = torch.bincount(group_index, minlength=group_count).to(dtype=per_example_losses.dtype)
    group_means = group_sums / group_sizes
    if reduction == "none":
        return group_means
    if reduction == "sum":
        return group_means.sum()
    return group_means.max()
