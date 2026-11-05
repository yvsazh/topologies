import math
import numbers

import torch
import torch.nn.functional as F


def jtt(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if set(params) - {"error_mask", "lambda_up"}:
        raise ValueError("unsupported JTT parameter")
    if "error_mask" not in params:
        raise ValueError("error_mask is required")
    if not isinstance(logits_or_prediction, torch.Tensor) or not logits_or_prediction.is_floating_point():
        raise ValueError("logits_or_prediction must be a floating point tensor")
    if logits_or_prediction.ndim != 2 or logits_or_prediction.shape[0] < 1 or logits_or_prediction.shape[1] < 1:
        raise ValueError("logits_or_prediction must have shape [B, C] with positive dimensions")
    if not isinstance(targets, torch.Tensor) or targets.ndim != 1:
        raise ValueError("targets must be a one-dimensional tensor of class indices")
    if tuple(targets.shape) != (logits_or_prediction.shape[0],):
        raise ValueError("targets must have shape [B] matching logits_or_prediction")
    if targets.device != logits_or_prediction.device:
        raise ValueError("targets and logits_or_prediction must be on the same device")
    if targets.dtype == torch.bool or targets.is_complex():
        raise ValueError("targets must contain integer class indices")
    if targets.is_floating_point():
        if not bool(torch.isfinite(targets).all()) or not bool((targets == targets.round()).all()):
            raise ValueError("targets must contain finite integer class indices")
    elif targets.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("targets must contain integer class indices")
    if ignore_index is not None and (isinstance(ignore_index, bool) or not isinstance(ignore_index, numbers.Integral)):
        raise ValueError("ignore_index must be an integer or None")
    if not isinstance(reduction, str) or reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    error_mask = params["error_mask"]
    if not isinstance(error_mask, torch.Tensor) or tuple(error_mask.shape) != (logits_or_prediction.shape[0],):
        raise ValueError("error_mask must be a tensor with shape [B]")
    if error_mask.device != logits_or_prediction.device:
        raise ValueError("error_mask and logits_or_prediction must be on the same device")
    if error_mask.is_complex():
        raise ValueError("error_mask must be boolean or binary numeric values")
    if error_mask.is_floating_point() and not bool(torch.isfinite(error_mask).all()):
        raise ValueError("error_mask must contain finite binary values")
    if not bool(((error_mask == 0) | (error_mask == 1)).all()):
        raise ValueError("error_mask must contain only zero and one")
    lambda_up = params.get("lambda_up", 1.0)
    lambda_up_valid = (
        isinstance(lambda_up, numbers.Real)
        and not isinstance(lambda_up, bool)
        and math.isfinite(float(lambda_up))
        and lambda_up > 0
    )
    if not lambda_up_valid:
        raise ValueError("lambda_up must be a positive finite real number")
    class_count = logits_or_prediction.shape[1]
    class_targets = targets.to(dtype=torch.long)
    if ignore_index is None:
        valid = torch.ones_like(class_targets, dtype=torch.bool)
    else:
        valid = class_targets != int(ignore_index)
    invalid = valid & ((class_targets < 0) | (class_targets >= class_count))
    if bool(invalid.any()):
        raise ValueError("targets contain a class index outside the logits class range")
    safe_targets = class_targets.masked_fill(~valid, 0)
    losses = F.cross_entropy(logits_or_prediction, safe_targets, reduction="none")
    weights = torch.where(
        error_mask.to(dtype=torch.bool),
        logits_or_prediction.new_tensor(float(lambda_up)),
        logits_or_prediction.new_tensor(1.0),
    )
    losses = (losses * weights).masked_fill(~valid, 0)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.sum() / valid.sum().to(dtype=losses.dtype)
