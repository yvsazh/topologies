import math

import torch


def proxy_anchor(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    unknown = set(params) - {"proxies", "alpha", "delta"}
    if unknown:
        raise ValueError(f"unsupported parameters: {', '.join(sorted(unknown))}")
    if "proxies" not in params:
        raise ValueError("proxies is required")

    proxies = params["proxies"]
    alpha = params.get("alpha", 32.0)
    delta = params.get("delta", 0.1)
    alpha_valid = not isinstance(alpha, bool) and isinstance(alpha, (int, float))
    try:
        alpha_valid = alpha_valid and math.isfinite(alpha)
    except OverflowError:
        alpha_valid = False
    if not alpha_valid or alpha <= 0:
        raise ValueError("alpha must be a finite positive scalar")
    delta_valid = not isinstance(delta, bool) and isinstance(delta, (int, float))
    try:
        delta_valid = delta_valid and math.isfinite(delta)
    except OverflowError:
        delta_valid = False
    if not delta_valid or delta < 0:
        raise ValueError("delta must be a finite nonnegative scalar")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if ignore_index is not None and (isinstance(ignore_index, bool) or not isinstance(ignore_index, int)):
        raise ValueError("ignore_index must be an integer or None")

    if not isinstance(logits_or_prediction, torch.Tensor) or logits_or_prediction.ndim != 2:
        raise ValueError("logits_or_prediction must be a tensor with shape [B, D]")
    if not isinstance(proxies, torch.Tensor) or proxies.ndim != 2:
        raise ValueError("proxies must be a tensor with shape [C, D]")
    embeddings = logits_or_prediction
    if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        raise ValueError("embeddings must have nonempty batch and feature dimensions")
    if proxies.shape[0] == 0 or proxies.shape[1] == 0:
        raise ValueError("proxies must have nonempty class and feature dimensions")
    if not embeddings.is_floating_point() or not proxies.is_floating_point():
        raise ValueError("embeddings and proxies must have floating-point dtypes")
    if embeddings.shape[1] != proxies.shape[1]:
        raise ValueError("embeddings and proxies must have the same feature dimension")
    if embeddings.device != proxies.device or embeddings.dtype != proxies.dtype:
        raise ValueError("embeddings and proxies must have the same device and dtype")
    if not torch.isfinite(embeddings).all() or not torch.isfinite(proxies).all():
        raise ValueError("embeddings and proxies must be finite")
    if not isinstance(targets, torch.Tensor) or targets.ndim != 1 or targets.shape[0] != embeddings.shape[0]:
        raise ValueError("targets must be a tensor with shape [B]")
    if targets.device != embeddings.device:
        raise ValueError("targets and embeddings must be on the same device")
    if targets.is_complex() or targets.dtype == torch.bool:
        raise ValueError("targets must contain integer class IDs")
    if targets.is_floating_point() and (
        not torch.isfinite(targets).all() or not torch.equal(targets, targets.floor())
    ):
        raise ValueError("floating-point targets must contain finite integer class IDs")

    if ignore_index is not None:
        keep = targets != ignore_index
        embeddings = embeddings[keep]
        targets = targets[keep]
    if targets.numel() == 0:
        raise ValueError("no samples remain after applying ignore_index")
    if torch.any(targets < 0) or torch.any(targets >= proxies.shape[0]):
        raise ValueError("targets must be class IDs in [0, C) after ignoring samples")

    labels = targets.to(dtype=torch.long)
    similarities = torch.nn.functional.normalize(embeddings, p=2, dim=-1) @ torch.nn.functional.normalize(
        proxies, p=2, dim=-1
    ).transpose(0, 1)
    class_count = proxies.shape[0]
    class_indices = torch.arange(class_count, device=labels.device)
    positive_mask = labels[:, None] == class_indices[None, :]
    negative_mask = ~positive_mask
    positive_logits = -alpha * (similarities - delta)
    negative_logits = alpha * (similarities + delta)
    baseline = similarities.new_zeros((1, class_count))
    positive_terms = torch.logsumexp(
        torch.cat((positive_logits.masked_fill(~positive_mask, -torch.inf), baseline), dim=0), dim=0
    )
    negative_terms = torch.logsumexp(
        torch.cat((negative_logits.masked_fill(~negative_mask, -torch.inf), baseline), dim=0), dim=0
    )
    positive_classes = positive_mask.any(dim=0)
    positive_scale = class_count / positive_classes.sum().to(dtype=similarities.dtype)
    per_proxy = positive_terms * positive_scale + negative_terms

    if reduction == "none":
        return per_proxy
    if reduction == "sum":
        return per_proxy.sum()
    return per_proxy.mean()
