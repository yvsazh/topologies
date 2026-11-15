import torch


def sphereface(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if not isinstance(logits_or_prediction, torch.Tensor) or not logits_or_prediction.is_floating_point():
        raise ValueError("logits_or_prediction must be a floating-point tensor")
    logits = logits_or_prediction
    if logits.ndim != 2 or logits.shape[0] < 1 or logits.shape[1] < 2:
        raise ValueError("logits_or_prediction must have shape [B, K] with B >= 1 and K >= 2")
    if not isinstance(targets, torch.Tensor) or targets.dtype != torch.long or targets.ndim != 1:
        raise ValueError("targets must be a long tensor with shape [B]")
    if targets.shape[0] != logits.shape[0]:
        raise ValueError("targets must have the same batch size as logits_or_prediction")
    if targets.device != logits.device:
        raise ValueError("targets and logits_or_prediction must be on the same device")
    if ignore_index is not None and (not isinstance(ignore_index, int) or isinstance(ignore_index, bool)):
        raise ValueError("ignore_index must be an integer or None")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if "feature_norm" not in params:
        raise ValueError("feature_norm is required")
    feature_norm = params.pop("feature_norm")
    m = params.pop("m", 4)
    if params:
        raise ValueError("unsupported parameters: " + ", ".join(sorted(params)))
    if type(m) is not int or m < 1:
        raise ValueError("m must be a positive integer")
    if not isinstance(feature_norm, torch.Tensor) or not feature_norm.is_floating_point():
        raise ValueError("feature_norm must be a floating-point tensor")
    if feature_norm.shape not in ((logits.shape[0],), (logits.shape[0], 1)):
        raise ValueError("feature_norm must have shape [B] or [B, 1]")
    if feature_norm.device != logits.device or feature_norm.dtype != logits.dtype:
        raise ValueError("feature_norm must have the same dtype and device as logits_or_prediction")
    feature_norm = feature_norm.reshape(logits.shape[0])
    if not torch.isfinite(logits).all() or not torch.isfinite(feature_norm).all():
        raise ValueError("logits_or_prediction and feature_norm must contain only finite values")
    if not (feature_norm > 0).all():
        raise ValueError("feature_norm values must be positive")
    tolerance = 8 * torch.finfo(logits.dtype).eps
    if (logits.abs() > feature_norm.unsqueeze(1) * (1 + tolerance)).any():
        raise ValueError("absolute class scores must not exceed feature_norm")
    valid = torch.ones_like(targets, dtype=torch.bool) if ignore_index is None else targets.ne(ignore_index)
    if ((targets[valid] < 0) | (targets[valid] >= logits.shape[1])).any():
        raise ValueError("non-ignored targets must be valid class indices")
    safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
    target_scores = logits.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
    cosine = (target_scores / feature_norm).clamp(-1, 1)
    theta = torch.acos(cosine)
    segment = torch.floor(m * theta / torch.pi).to(torch.long).clamp(0, m - 1)
    segment_value = segment.to(logits.dtype)
    sign = torch.where(segment.remainder(2) == 0, 1.0, -1.0).to(logits.dtype)
    angular_score = sign * torch.cos(m * theta) - 2 * segment_value
    corrected_target = feature_norm * angular_score
    class_indices = torch.arange(logits.shape[1], device=logits.device)
    target_mask = class_indices.unsqueeze(0).eq(safe_targets.unsqueeze(1))
    modified_logits = torch.where(target_mask, corrected_target.unsqueeze(1), logits)
    per_example = torch.logsumexp(modified_logits, dim=1) - modified_logits.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
    per_example = torch.where(valid, per_example, torch.zeros_like(per_example))
    if reduction == "none":
        return per_example
    if reduction == "sum":
        return per_example.sum()
    return per_example.sum() / valid.sum().clamp_min(1)
