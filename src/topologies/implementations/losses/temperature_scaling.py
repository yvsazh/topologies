import math
import numbers

import torch


def temperature_scaling(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if set(params) - {"temperature"}:
        raise ValueError("unsupported parameter; only temperature is supported")
    if not isinstance(logits_or_prediction, torch.Tensor) or not logits_or_prediction.is_floating_point():
        raise ValueError("logits_or_prediction must be a floating point tensor")
    temperature = params.get("temperature", 1.0)
    if isinstance(temperature, torch.Tensor):
        temperature_valid = temperature.ndim == 0 and temperature.dtype != torch.bool and not temperature.is_complex()
        if temperature_valid:
            temperature_value = temperature.detach().item()
            temperature_valid = math.isfinite(float(temperature_value)) and temperature_value > 0
    else:
        temperature_valid = isinstance(temperature, numbers.Real) and not isinstance(temperature, bool)
        if temperature_valid:
            temperature_valid = math.isfinite(float(temperature)) and temperature > 0
    if not temperature_valid:
        raise ValueError("temperature must be a positive finite scalar")
    if isinstance(temperature, torch.Tensor):
        divisor = temperature.to(device=logits_or_prediction.device, dtype=logits_or_prediction.dtype)
    else:
        divisor = float(temperature)
    if logits_or_prediction.ndim != 2 or logits_or_prediction.shape[1] < 1:
        raise ValueError("logits_or_prediction must have shape [B, K] with at least one class")
    if not isinstance(targets, torch.Tensor) or targets.dtype != torch.long:
        raise ValueError("targets must be a long tensor of class indices")
    if tuple(targets.shape) != (logits_or_prediction.shape[0],):
        raise ValueError("targets must have shape [B] matching logits_or_prediction")
    if targets.device != logits_or_prediction.device:
        raise ValueError("targets and logits_or_prediction must be on the same device")
    if ignore_index is not None and (isinstance(ignore_index, bool) or not isinstance(ignore_index, numbers.Integral)):
        raise ValueError("ignore_index must be an integer or None")
    if not isinstance(reduction, str) or reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if ignore_index is None:
        valid = torch.ones_like(targets, dtype=torch.bool)
    else:
        valid = targets != int(ignore_index)
    class_count = logits_or_prediction.shape[1]
    invalid = valid & ((targets < 0) | (targets >= class_count))
    if bool(invalid.any()):
        raise ValueError("targets contain a class index outside the logits class range")
    safe_targets = targets.masked_fill(~valid, 0)
    log_probabilities = torch.log_softmax(logits_or_prediction / divisor, dim=1)
    losses = -log_probabilities.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
    losses = losses.masked_fill(~valid, 0)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.sum() / valid.sum().to(dtype=losses.dtype)
