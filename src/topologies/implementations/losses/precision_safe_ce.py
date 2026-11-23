import torch


def precision_safe_ce(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    value_error = ValueError

    def stable_log_softmax(logits):
        maximum, maximum_index = logits.max(dim=-1, keepdim=True)
        class_indices = torch.arange(logits.shape[-1], device=logits.device)
        maximum_mask = class_indices.eq(maximum_index)
        shifted = torch.where(maximum_mask, torch.zeros_like(logits), logits - maximum)
        other_exp_sum = torch.where(maximum_mask, torch.zeros_like(logits), torch.exp(shifted)).sum(dim=-1, keepdim=True)
        return shifted - torch.log1p(other_exp_sum)

    def reduce(per_example, valid=None):
        if reduction == "none":
            return per_example
        total = per_example.sum()
        if reduction == "sum":
            return total
        denominator = per_example.numel() if valid is None else valid.sum()
        if isinstance(denominator, int):
            denominator = max(denominator, 1)
        else:
            denominator = denominator.clamp_min(1)
        return total / denominator

    if params:
        raise ValueError("unsupported parameters: " + ", ".join(sorted(params)))
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if ignore_index is not None and (not isinstance(ignore_index, int) or isinstance(ignore_index, bool)):
        raise ValueError("ignore_index must be an integer or None")
    if not isinstance(logits_or_prediction, torch.Tensor) or not logits_or_prediction.is_floating_point():
        raise ValueError("logits_or_prediction must be a floating-point tensor")
    if not isinstance(targets, torch.Tensor):
        raise value_error("targets must be a tensor")
    if logits_or_prediction.layout != torch.strided or targets.layout != torch.strided:
        raise ValueError("logits_or_prediction and targets must be dense tensors")
    if logits_or_prediction.ndim < 1 or logits_or_prediction.shape[-1] < 1:
        raise ValueError("logits_or_prediction must have shape [..., C] with C >= 1")
    if logits_or_prediction.device != targets.device:
        raise ValueError("logits_or_prediction and targets must be on the same device")

    logits = logits_or_prediction
    if targets.is_floating_point():
        if targets.shape != logits.shape:
            raise ValueError("floating targets must have the same shape as logits_or_prediction")
        if targets.dtype != logits.dtype:
            raise ValueError("floating targets must have the same dtype as logits_or_prediction")
        if ignore_index is not None:
            raise ValueError("ignore_index must be None for floating teacher-logit targets")
        if torch.isnan(logits).any() or torch.isposinf(logits).any() or torch.isnan(targets).any() or torch.isposinf(targets).any():
            raise ValueError("logits and teacher logits must not contain NaN or positive infinity")
        if not torch.isfinite(logits).any(dim=-1).all() or not torch.isfinite(targets).any(dim=-1).all():
            raise ValueError("each logits row must contain at least one finite value")

        log_q = stable_log_softmax(logits)
        log_p = stable_log_softmax(targets)
        q = torch.exp(log_q)
        p = torch.exp(log_p)
        finite_pair = torch.isfinite(logits) & torch.isfinite(targets)
        delta = torch.where(finite_pair, logits - targets, torch.zeros_like(logits))
        centered = delta - (p * delta).sum(dim=-1, keepdim=True)
        normalizer = torch.log1p((p * torch.expm1(centered)).sum(dim=-1, keepdim=True))
        log_ratio = centered - normalizer
        reconstruct = (
            torch.isfinite(logits).all(dim=-1, keepdim=True)
            & torch.isfinite(targets).all(dim=-1, keepdim=True)
            & torch.isfinite(log_ratio)
            & (log_ratio.abs() < 1)
        )
        reconstructed_difference = p * torch.expm1(log_ratio)
        probability_difference = torch.where(reconstruct, reconstructed_difference, q - p)
        gradient = probability_difference - q * probability_difference.sum(dim=-1, keepdim=True)
        safe_log_q = torch.where(p > 0, log_q, torch.zeros_like(log_q)).detach()
        forward_loss = -(p * safe_log_q).sum(dim=-1)
        finite_logits = torch.isfinite(logits)
        zero_value_proxy = torch.where(finite_logits, logits - logits.detach(), torch.zeros_like(logits))
        per_example = forward_loss + (gradient.detach() * zero_value_proxy).sum(dim=-1)
        return reduce(per_example)

    if targets.is_complex() or targets.dtype == torch.bool:
        raise ValueError("targets must contain integer class indices or floating teacher logits")
    if targets.shape != logits.shape[:-1]:
        raise ValueError("integer targets must match the leading logits dimensions")
    valid = torch.ones_like(targets, dtype=torch.bool) if ignore_index is None else targets.ne(ignore_index)
    integer_targets = targets.to(dtype=torch.long)
    if ((integer_targets[valid] < 0) | (integer_targets[valid] >= logits.shape[-1])).any():
        raise ValueError("non-ignored targets must be valid class indices")

    work_logits = torch.where(valid.unsqueeze(-1), logits, torch.zeros_like(logits))
    if torch.isnan(work_logits[valid]).any() or torch.isposinf(work_logits[valid]).any():
        raise ValueError("valid logits must not contain NaN or positive infinity")
    safe_targets = torch.where(valid, integer_targets, torch.zeros_like(integer_targets))
    target_logits = work_logits.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    if not torch.isfinite(target_logits[valid]).all():
        raise ValueError("non-ignored target logits must be finite")
    target_mask = torch.arange(logits.shape[-1], device=logits.device).eq(safe_targets.unsqueeze(-1))
    shifted = torch.where(target_mask, torch.zeros_like(work_logits), work_logits - target_logits.unsqueeze(-1))
    maximum = shifted.max(dim=-1).values
    other_exp_sum = torch.where(
        target_mask,
        torch.zeros_like(shifted),
        torch.exp(shifted - maximum.unsqueeze(-1)),
    ).sum(dim=-1)
    per_example = maximum + torch.log1p(other_exp_sum + torch.expm1(-maximum))
    per_example = torch.where(valid, per_example, torch.zeros_like(per_example))
    return reduce(per_example, valid)
