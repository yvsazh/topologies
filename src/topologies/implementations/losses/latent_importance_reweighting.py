import torch


def latent_importance_reweighting(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if ignore_index is not None:
        raise ValueError("latent_importance_reweighting does not support ignore_index")
    if not isinstance(reduction, str) or reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if "importance_weights" not in params:
        raise ValueError("importance_weights is required")
    unsupported = set(params) - {"importance_weights", "mask"}
    if unsupported:
        raise ValueError(f"unsupported parameters: {sorted(unsupported)}")
    if not torch.is_tensor(logits_or_prediction) or not torch.is_tensor(targets):
        raise ValueError("prediction and targets must be tensors")
    if not logits_or_prediction.is_floating_point() or not targets.is_floating_point():
        raise ValueError("prediction and targets must have floating-point dtypes")
    if logits_or_prediction.ndim < 1 or any(size < 1 for size in logits_or_prediction.shape):
        raise ValueError("prediction must have a nonempty shape [..., D]")
    if targets.shape != logits_or_prediction.shape:
        raise ValueError("prediction and targets must have identical shapes")
    if logits_or_prediction.dtype != targets.dtype or logits_or_prediction.device != targets.device:
        raise ValueError("prediction and targets must have matching dtype and device")
    if not torch.isfinite(logits_or_prediction).all() or not torch.isfinite(targets).all():
        raise ValueError("prediction and targets must contain only finite values")

    try:
        raw_weights = torch.as_tensor(params["importance_weights"])
    except (TypeError, ValueError, RuntimeError, OverflowError) as error:
        raise ValueError("importance_weights must be a finite real vector") from error
    if raw_weights.dtype == torch.bool or raw_weights.is_complex():
        raise ValueError("importance_weights must be a finite real vector")
    weights = raw_weights.to(device=logits_or_prediction.device, dtype=logits_or_prediction.dtype)
    if weights.ndim != 1 or weights.shape[0] != logits_or_prediction.shape[-1]:
        raise ValueError("importance_weights must have shape [D] matching the final dimension")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("importance_weights must be finite and nonnegative")
    if not torch.isclose(weights.sum(), weights.new_tensor(1.0), rtol=1e-5, atol=1e-6):
        raise ValueError("importance_weights must sum to one")

    raw_mask = params.get("mask")
    if raw_mask is None:
        mask = torch.ones(
            logits_or_prediction.shape[:-1],
            dtype=torch.bool,
            device=logits_or_prediction.device,
        )
    else:
        if not torch.is_tensor(raw_mask):
            raise ValueError("mask must be a boolean or binary tensor")
        if raw_mask.shape != logits_or_prediction.shape[:-1]:
            raise ValueError("mask must match the prediction's position dimensions")
        if raw_mask.device != logits_or_prediction.device:
            raise ValueError("mask and prediction must have matching devices")
        if raw_mask.dtype != torch.bool:
            if not (raw_mask.is_floating_point() or raw_mask.dtype in {
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            }):
                raise ValueError("mask must be a boolean or binary tensor")
            if not torch.isfinite(raw_mask).all() or ((raw_mask != 0) & (raw_mask != 1)).any():
                raise ValueError("mask values must be finite and binary")
            mask = raw_mask.to(dtype=torch.bool)
        else:
            mask = raw_mask

    batch_size = logits_or_prediction.shape[0] if logits_or_prediction.ndim > 1 else 1
    flat_mask = mask.reshape(batch_size, -1)
    valid_positions = flat_mask.sum(dim=-1)
    if (valid_positions == 0).any():
        raise ValueError("mask must include at least one valid position per graph")

    difference = (logits_or_prediction - targets) * mask.unsqueeze(-1).to(
        dtype=logits_or_prediction.dtype
    )
    flat_difference = difference.reshape(batch_size, -1, logits_or_prediction.shape[-1])
    dimension_norms = torch.linalg.vector_norm(flat_difference, ord=2, dim=1)
    graph_losses = (dimension_norms * weights).sum(dim=-1) / logits_or_prediction.shape[-1]
    if reduction == "none":
        return graph_losses
    if reduction == "sum":
        return graph_losses.sum()
    return graph_losses.mean()
