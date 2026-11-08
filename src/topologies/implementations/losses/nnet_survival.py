import torch
import torch.nn.functional as F


def nnet_survival_loss(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    unsupported = set(params) - {"from_logits"}
    if unsupported:
        raise ValueError(f"unsupported parameters: {', '.join(sorted(unsupported))}")
    if ignore_index is not None:
        raise ValueError("ignore_index must be None for packed survival targets")
    if type(reduction) is not str or reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    from_logits = params.get("from_logits", False)
    if type(from_logits) is not bool:
        raise ValueError("from_logits must be a bool")
    if not isinstance(logits_or_prediction, torch.Tensor) or not logits_or_prediction.is_floating_point():
        raise ValueError("logits_or_prediction must be a floating point tensor")
    if logits_or_prediction.ndim != 2:
        raise ValueError("logits_or_prediction must have shape [B, T]")
    batch_size, n_intervals = logits_or_prediction.shape
    if batch_size < 1 or n_intervals < 1:
        raise ValueError("logits_or_prediction dimensions must be nonempty")
    if not isinstance(targets, torch.Tensor) or not targets.is_floating_point():
        raise ValueError("targets must be a floating point packed target tensor")
    if targets.ndim != 2 or targets.shape[0] != batch_size or targets.shape[1] != 2 * n_intervals:
        raise ValueError("targets must have shape [B, 2T] matching logits_or_prediction")
    if targets.dtype != logits_or_prediction.dtype or targets.device != logits_or_prediction.device:
        raise ValueError("targets and logits_or_prediction must have the same dtype and device")
    if not bool(torch.isfinite(logits_or_prediction).all()):
        raise ValueError("logits_or_prediction must contain only finite values")
    if not bool(torch.isfinite(targets).all()):
        raise ValueError("targets must contain only finite values")
    if bool(((targets != 0) & (targets != 1)).any()):
        raise ValueError("targets must contain only binary survival and event flags")
    if not from_logits and bool(((logits_or_prediction < 0) | (logits_or_prediction > 1)).any()):
        raise ValueError("survival probabilities must be in [0, 1]")

    survival_flags, event_flags = targets.split(n_intervals, dim=1)
    if from_logits:
        log_survival = -F.softplus(-logits_or_prediction)
        log_event_complement = -F.softplus(logits_or_prediction)
    else:
        epsilon = torch.finfo(logits_or_prediction.dtype).eps
        log_survival = torch.log(logits_or_prediction.clamp_min(epsilon))
        log_event_complement = torch.log((1 - logits_or_prediction).clamp_min(epsilon))
    per_person = -(survival_flags * log_survival + event_flags * log_event_complement).sum(dim=1)
    if reduction == "none":
        return per_person
    if reduction == "sum":
        return per_person.sum()
    return per_person.mean()
