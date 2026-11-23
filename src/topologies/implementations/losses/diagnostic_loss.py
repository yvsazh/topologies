import math

import torch


def diagnostic_loss(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    allowed = {
        "report_mask",
        "diagnostic_token_ids",
        "diagnostic_token_mask",
        "diagnostic_concept_mask",
        "diagnostic_positive_mask",
        "diagnostic_concept_weights",
        "sample_weights",
        "lambda_pos",
        "lambda_neg",
        "tau",
        "epsilon",
    }
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(f"unsupported parameters: {', '.join(sorted(unknown))}")
    if not isinstance(logits_or_prediction, torch.Tensor) or logits_or_prediction.ndim != 3:
        raise ValueError("logits_or_prediction must be a tensor with shape [B, T, V]")
    if not logits_or_prediction.is_floating_point():
        raise ValueError("logits_or_prediction must have a floating-point dtype")
    batch, positions, vocabulary = logits_or_prediction.shape
    if batch < 1 or positions < 1 or vocabulary < 1:
        raise ValueError("logits_or_prediction dimensions must be nonzero")
    if not isinstance(targets, torch.Tensor) or targets.shape != (batch, positions):
        raise ValueError("targets must have shape [B, T] matching logits_or_prediction")
    if targets.dtype == torch.bool:
        raise ValueError("targets must contain integer vocabulary IDs")
    labels = targets.to(device=logits_or_prediction.device)
    if labels.is_complex():
        raise ValueError("targets must contain integer vocabulary IDs")
    if labels.is_floating_point() and (
        not bool(torch.isfinite(labels).all().item()) or not bool((labels == labels.round()).all().item())
    ):
        raise ValueError("targets must contain finite integer vocabulary IDs")
    labels = labels.long()
    if ignore_index is not None and (not isinstance(ignore_index, int) or isinstance(ignore_index, bool)):
        raise ValueError("ignore_index must be an integer or None")
    if not isinstance(reduction, str) or reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")

    for name in ("report_mask", "diagnostic_token_ids", "diagnostic_positive_mask"):
        if name not in params:
            raise ValueError(f"{name} is required")
    report_mask = _mask(params["report_mask"], (batch, positions), "report_mask", labels.device)
    valid_positions = report_mask
    if ignore_index is not None:
        valid_positions = valid_positions & (labels != ignore_index)
    invalid_labels = valid_positions & ((labels < 0) | (labels >= vocabulary))
    if bool(invalid_labels.any().item()):
        raise ValueError("a valid report label is outside the logits vocabulary")
    if not bool(valid_positions.any().item()):
        raise ValueError("at least one valid report token is required")

    lambda_pos = _finite_scalar(params.get("lambda_pos", 0.001), "lambda_pos")
    lambda_neg = _finite_scalar(params.get("lambda_neg", 0.00025), "lambda_neg")
    tau = _finite_scalar(params.get("tau", 8.0), "tau")
    epsilon = _finite_scalar(params.get("epsilon", 1e-6), "epsilon")
    if lambda_pos < 0 or lambda_neg < 0:
        raise ValueError("lambda_pos and lambda_neg must be nonnegative")
    if tau <= 0:
        raise ValueError("tau must be positive")
    if not 0 < epsilon < 0.5:
        raise ValueError("epsilon must be in (0, 0.5)")

    logits = logits_or_prediction
    if logits.dtype in {torch.float16, torch.bfloat16}:
        logits = logits.float()
    if epsilon < torch.finfo(logits.dtype).eps:
        raise ValueError("epsilon must be at least the computation dtype machine epsilon")
    safe_logits = logits.masked_fill(~valid_positions.unsqueeze(-1), 0.0)
    if not bool(torch.isfinite(safe_logits[valid_positions]).all().item()):
        raise ValueError("logits at valid report positions must be finite")
    log_normalizer = torch.logsumexp(safe_logits, dim=-1)
    safe_labels = labels.masked_fill(~valid_positions, 0)
    label_logits = safe_logits.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_losses = (log_normalizer - label_logits).masked_fill(~valid_positions, 0.0)
    token_counts = valid_positions.sum(dim=1)
    cross_entropy_per_sample = token_losses.sum(dim=1) / token_counts.clamp_min(1)
    cross_entropy = token_losses.sum() / token_counts.sum()

    ids = _integer_tensor(
        params["diagnostic_token_ids"], "diagnostic_token_ids", logits.device
    )
    if ids.ndim != 3 or ids.shape[0] != batch:
        raise ValueError("diagnostic_token_ids must have shape [B, D, K]")
    _, concepts, token_count = ids.shape
    token_mask = _mask(
        params.get("diagnostic_token_mask", torch.ones_like(ids, dtype=torch.bool)),
        (batch, concepts, token_count),
        "diagnostic_token_mask",
        logits.device,
    )
    concept_mask = _mask(
        params.get("diagnostic_concept_mask", torch.ones(batch, concepts, dtype=torch.bool, device=logits.device)),
        (batch, concepts),
        "diagnostic_concept_mask",
        logits.device,
    )
    positive_mask = _mask(
        params["diagnostic_positive_mask"],
        (batch, concepts),
        "diagnostic_positive_mask",
        logits.device,
    )
    token_mask = token_mask & concept_mask.unsqueeze(-1)
    if bool((concept_mask & ~token_mask.any(dim=-1)).any().item()):
        raise ValueError("each active diagnostic concept must have at least one token ID")
    if bool((token_mask & ((ids < 0) | (ids >= vocabulary))).any().item()):
        raise ValueError("a valid diagnostic token ID is outside the logits vocabulary")
    if token_count > 1:
        duplicate_ids = (ids.unsqueeze(-1) == ids.unsqueeze(-2)) & token_mask.unsqueeze(-1) & token_mask.unsqueeze(-2)
        duplicate_ids = duplicate_ids & ~torch.eye(token_count, dtype=torch.bool, device=logits.device)
        if bool(duplicate_ids.any().item()):
            raise ValueError("diagnostic token IDs in each concept must be unique")

    dtype = logits.dtype
    concept_weights = params.get("diagnostic_concept_weights")
    if concept_weights is None:
        concept_weights = torch.ones(batch, concepts, dtype=dtype, device=logits.device)
    else:
        try:
            concept_weights = torch.as_tensor(concept_weights, device=logits.device, dtype=dtype)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError("diagnostic_concept_weights must be a finite nonnegative tensor") from error
    if concept_weights.shape != (batch, concepts):
        raise ValueError("diagnostic_concept_weights must have shape [B, D]")
    if not bool(torch.isfinite(concept_weights).all().item()) or bool((concept_weights < 0).any().item()):
        raise ValueError("diagnostic_concept_weights must be finite and nonnegative")
    sample_weights = params.get("sample_weights")
    if sample_weights is None:
        sample_weights = torch.ones(batch, dtype=dtype, device=logits.device)
    else:
        try:
            sample_weights = torch.as_tensor(sample_weights, device=logits.device, dtype=dtype)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError("sample_weights must be a finite nonnegative tensor") from error
    if sample_weights.shape != (batch,):
        raise ValueError("sample_weights must have shape [B]")
    if not bool(torch.isfinite(sample_weights).all().item()) or bool((sample_weights < 0).any().item()):
        raise ValueError("sample_weights must be finite and nonnegative")

    active_samples = concept_mask.any(dim=1)
    if bool((active_samples & ~valid_positions.any(dim=1)).any().item()):
        raise ValueError("samples with diagnostic concepts need a valid report position")
    diagnostic_per_sample = logits.new_zeros(batch)
    if concepts > 0:
        safe_ids = ids.masked_fill(~token_mask, 0)
        log_probabilities = safe_logits - log_normalizer.unsqueeze(-1)
        selected = log_probabilities.gather(
            2,
            safe_ids.flatten(1)[:, None, :].expand(batch, positions, concepts * token_count),
        ).reshape(batch, positions, concepts, token_count).exp()
        mass = (selected * token_mask[:, None, :, :]).sum(dim=-1)
        positive_mass = mass.clamp(max=1.0 - epsilon).masked_fill(~valid_positions[:, :, None], 0.0)
        log_none = torch.log1p(-positive_mass).sum(dim=1)
        positive_occurrence = -torch.expm1(log_none)
        positive_loss = -torch.log(positive_occurrence.clamp_min(epsilon))
        lse_positions = valid_positions.clone()
        lse_positions[:, 0] |= ~valid_positions.any(dim=1)
        negative_mass = mass.clamp_min(epsilon).log()
        negative_loss = torch.logsumexp(
            (tau * negative_mass).masked_fill(~lse_positions[:, :, None], -torch.inf), dim=1
        ) / tau

        positive_weights = concept_weights * concept_mask * positive_mask
        negative_weights = concept_weights * concept_mask * ~positive_mask
        positive_denominator = positive_weights.sum(dim=1)
        negative_denominator = negative_weights.sum(dim=1)
        positive_per_sample = (positive_loss * positive_weights).sum(dim=1) / torch.where(
            positive_denominator > 0, positive_denominator, torch.ones_like(positive_denominator)
        )
        negative_per_sample = (negative_loss * negative_weights).sum(dim=1) / torch.where(
            negative_denominator > 0, negative_denominator, torch.ones_like(negative_denominator)
        )
        diagnostic_per_sample = sample_weights * (
            lambda_pos * positive_per_sample + lambda_neg * negative_per_sample
        )

    if reduction == "none":
        return cross_entropy_per_sample + diagnostic_per_sample
    if reduction == "sum":
        return (cross_entropy_per_sample + diagnostic_per_sample).sum()
    active_count = active_samples.sum().clamp_min(1)
    diagnostic = diagnostic_per_sample.sum() / active_count
    return cross_entropy + diagnostic


def _finite_scalar(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        result = math.nan
    else:
        try:
            result = float(value)
        except (OverflowError, ValueError):
            result = math.nan
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _mask(value, shape, name, device):
    try:
        tensor = torch.as_tensor(value, device=device)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{name} must be a boolean or binary tensor") from error
    if tensor.shape != shape:
        raise ValueError(f"{name} must have shape {list(shape)}")
    if tensor.dtype != torch.bool:
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"{name} must contain only zero and one")
        if bool(((tensor != 0) & (tensor != 1)).any().item()):
            raise ValueError(f"{name} must contain only zero and one")
    return tensor.bool()


def _integer_tensor(value, name, device):
    try:
        tensor = torch.as_tensor(value, device=device)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{name} must contain integer IDs") from error
    if tensor.dtype == torch.bool:
        raise ValueError(f"{name} must contain integer IDs")
    if tensor.is_complex():
        raise ValueError(f"{name} must contain integer IDs")
    if tensor.is_floating_point() and (
        not bool(torch.isfinite(tensor).all().item()) or not bool((tensor == tensor.round()).all().item())
    ):
        raise ValueError(f"{name} must contain finite integer IDs")
    return tensor.long()
