import torch


def lovasz_softmax(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if set(params) - {"classes", "mask"}:
        raise ValueError("unsupported parameter; only classes and mask are supported")
    if ignore_index is not None:
        raise ValueError("ignore_index must be None because ignored pixels are unsupported")
    if not isinstance(logits_or_prediction, torch.Tensor) or not logits_or_prediction.is_floating_point():
        raise ValueError("logits_or_prediction must be a floating point tensor")
    if logits_or_prediction.ndim < 3 or any(size < 1 for size in logits_or_prediction.shape):
        raise ValueError("logits_or_prediction must have nonempty shape [N, C, *S]")
    if not isinstance(targets, torch.Tensor) or targets.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise ValueError("targets must be an integer tensor of class indices")
    expected_shape = (logits_or_prediction.shape[0], *logits_or_prediction.shape[2:])
    if tuple(targets.shape) != expected_shape:
        raise ValueError("targets shape must match logits_or_prediction batch and spatial dimensions")
    if targets.device != logits_or_prediction.device:
        raise ValueError("targets and logits_or_prediction must be on the same device")
    if not isinstance(reduction, str) or reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    classes = params.get("classes", "all")
    if classes not in ("all", "present"):
        raise ValueError("classes must be 'all' or 'present'")
    target_indices = targets.to(dtype=torch.long).reshape(-1)
    flat_logits = logits_or_prediction.movedim(1, -1).reshape(-1, logits_or_prediction.shape[1]).transpose(0, 1)
    valid_mask = params.get("mask")
    if valid_mask is not None:
        if not isinstance(valid_mask, torch.Tensor) or valid_mask.dtype != torch.bool:
            raise ValueError("mask must be a boolean tensor")
        if tuple(valid_mask.shape) != expected_shape:
            raise ValueError("mask shape must match targets")
        if valid_mask.device != logits_or_prediction.device:
            raise ValueError("mask and logits_or_prediction must be on the same device")
        valid_mask = valid_mask.reshape(-1)
        target_indices = target_indices[valid_mask]
        flat_logits = flat_logits[:, valid_mask]
    class_count = logits_or_prediction.shape[1]
    if target_indices.numel() and bool(((target_indices < 0) | (target_indices >= class_count)).any()):
        raise ValueError("targets contain a class index outside the logits class range")
    probabilities = torch.softmax(flat_logits, dim=0)
    if classes == "present":
        class_indices = torch.bincount(target_indices, minlength=class_count).nonzero().flatten().tolist()
    else:
        class_indices = range(class_count)
    class_losses = []
    for class_index in class_indices:
        foreground = target_indices == class_index
        errors = torch.where(foreground, 1 - probabilities[class_index], probabilities[class_index])
        sorted_errors, permutation = torch.sort(errors, descending=True)
        sorted_foreground = foreground[permutation].to(dtype=torch.long)
        foreground_count = sorted_foreground.sum()
        intersection = foreground_count - sorted_foreground.cumsum(0)
        union = foreground_count + (1 - sorted_foreground).cumsum(0)
        work_dtype = torch.float32 if sorted_errors.dtype in (torch.float16, torch.bfloat16) else sorted_errors.dtype
        union_values = union.to(dtype=work_dtype)
        intersection_values = intersection.to(dtype=work_dtype)
        jaccard_loss = torch.where(
            union > 0,
            1 - intersection_values / union_values.clamp_min(1),
            torch.zeros_like(union_values),
        )
        increments = torch.cat((jaccard_loss[:1], jaccard_loss[1:] - jaccard_loss[:-1]))
        class_losses.append(torch.sum(sorted_errors * increments.to(dtype=sorted_errors.dtype)))
    if class_losses:
        losses = torch.stack(class_losses)
    else:
        losses = (logits_or_prediction.sum() * 0).expand(0)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if losses.numel() == 0:
        return logits_or_prediction.sum() * 0
    return losses.mean()
