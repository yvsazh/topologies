import math
from numbers import Real

import torch


def irm(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if ignore_index is not None:
        raise ValueError("ignore_index must be None for squared-error IRM")
    if not isinstance(reduction, str) or reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    unknown = set(params) - {"environment_ids", "penalty_weight"}
    if unknown:
        raise ValueError(f"unexpected parameters: {sorted(unknown)}")
    if "environment_ids" not in params:
        raise ValueError("environment_ids is required")
    if not isinstance(logits_or_prediction, torch.Tensor) or not logits_or_prediction.is_floating_point():
        raise ValueError("logits_or_prediction must be a floating-point tensor")
    if logits_or_prediction.ndim < 1 or logits_or_prediction.shape[0] == 0 or logits_or_prediction.numel() == 0:
        raise ValueError("logits_or_prediction must have a nonempty first dimension and output")
    if not isinstance(targets, torch.Tensor) or not targets.is_floating_point():
        raise ValueError("targets must be a floating-point tensor")
    if targets.shape != logits_or_prediction.shape:
        raise ValueError("targets must have the same shape as logits_or_prediction")
    if targets.device != logits_or_prediction.device:
        raise ValueError("targets and logits_or_prediction must be on the same device")
    environment_ids = params["environment_ids"]
    if not isinstance(environment_ids, torch.Tensor) or environment_ids.ndim != 1:
        raise ValueError("environment_ids must be a one-dimensional numeric tensor with shape [N]")
    if environment_ids.shape[0] != logits_or_prediction.shape[0]:
        raise ValueError("environment_ids must contain one ID per sample")
    if environment_ids.dtype == torch.bool or environment_ids.is_complex():
        raise ValueError("environment_ids must contain numeric integer-valued IDs")
    if environment_ids.is_floating_point():
        if not bool(torch.isfinite(environment_ids).all()):
            raise ValueError("environment_ids must be finite")
        if not torch.equal(environment_ids, environment_ids.round()):
            raise ValueError("environment_ids must contain integer-valued IDs")
    else:
        try:
            torch.iinfo(environment_ids.dtype)
        except TypeError as error:
            raise ValueError("environment_ids must contain numeric integer-valued IDs") from error
    penalty_weight = params.get("penalty_weight", 1.0)
    if isinstance(penalty_weight, bool) or not isinstance(penalty_weight, Real):
        error = ValueError("penalty_weight must be a finite nonnegative real number")
        raise error
    try:
        penalty_weight = float(penalty_weight)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("penalty_weight must be a finite nonnegative real number") from error
    if not math.isfinite(penalty_weight) or penalty_weight < 0:
        raise ValueError("penalty_weight must be a finite nonnegative real number")
    environment_ids = environment_ids.to(device=logits_or_prediction.device)
    distinct_environments = torch.unique(environment_ids, sorted=True)
    if distinct_environments.numel() < 2:
        raise ValueError("environment_ids must contain at least two distinct environments")
    terms = []
    for environment in distinct_environments:
        mask = environment_ids == environment
        prediction = logits_or_prediction[mask]
        target = targets[mask]
        residual = prediction - target
        risk = residual.square().mean()
        scale_gradient = (2 * prediction * residual).mean()
        terms.append(risk + penalty_weight * scale_gradient.square())
    per_environment = torch.stack(terms)
    if reduction == "none":
        return per_environment
    if reduction == "sum":
        return per_environment.sum()
    return per_environment.mean()
