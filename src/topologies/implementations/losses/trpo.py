import torch


def trpo(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if ignore_index is not None:
        raise ValueError("ignore_index is not supported")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    unknown = set(params) - {"old_log_prob"}
    if unknown:
        raise ValueError(f"unexpected parameters: {sorted(unknown)}")
    if "old_log_prob" not in params:
        raise ValueError("missing required parameters: ['old_log_prob']")
    old_log_prob = params["old_log_prob"]
    tensors = {
        "logits_or_prediction": logits_or_prediction,
        "targets": targets,
        "old_log_prob": old_log_prob,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            error = ValueError(f"{name} must be a tensor")
            raise error
        if not tensor.is_floating_point():
            raise ValueError(f"{name} must have a floating-point dtype")
        if tensor.ndim != 1 or tensor.shape != logits_or_prediction.shape:
            raise ValueError("all inputs must have identical one-dimensional shapes [B]")
        if tensor.numel() == 0:
            raise ValueError("inputs must contain at least one sample")
        if tensor.device != logits_or_prediction.device:
            raise ValueError("all inputs must be on the same device")
    loss = -torch.exp(logits_or_prediction - old_log_prob.detach()) * targets.detach()
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()
