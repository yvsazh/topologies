import math
import numbers

import torch


def simpo_loss(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if set(params) - {"beta", "gamma"}:
        raise ValueError("unsupported parameters; only beta and gamma are supported")
    if "beta" not in params or "gamma" not in params:
        raise ValueError("beta and gamma are required")
    if not isinstance(logits_or_prediction, torch.Tensor) or not logits_or_prediction.is_floating_point():
        raise ValueError("logits_or_prediction must be a floating point tensor")
    if logits_or_prediction.ndim != 4 or logits_or_prediction.shape[1] != 2:
        raise ValueError("logits_or_prediction must have shape [B, 2, T, V]")
    batch_size, _, sequence_length, vocabulary_size = logits_or_prediction.shape
    if batch_size < 1 or sequence_length < 1 or vocabulary_size < 1:
        raise ValueError("logits_or_prediction dimensions B, T, and V must be positive")
    if not isinstance(targets, torch.Tensor):
        error_type = ValueError
        raise error_type("targets must be an integer or integral-valued floating point tensor")
    if targets.dtype == torch.bool or targets.is_complex():
        raise ValueError("targets must contain integer token IDs")
    if tuple(targets.shape) != (batch_size, 2, sequence_length):
        raise ValueError("targets must have shape [B, 2, T] matching logits_or_prediction")
    if targets.device != logits_or_prediction.device:
        raise ValueError("targets and logits_or_prediction must be on the same device")
    if targets.is_floating_point() and (
        not bool(torch.isfinite(targets).all()) or not bool((targets == targets.trunc()).all())
    ):
        raise ValueError("floating point targets must contain finite integral token IDs")
    if ignore_index is not None and (
        isinstance(ignore_index, bool) or not isinstance(ignore_index, numbers.Integral)
    ):
        raise ValueError("ignore_index must be an integer or None")
    if not isinstance(reduction, str) or reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    beta = params["beta"]
    gamma = params["gamma"]
    if (
        isinstance(beta, bool)
        or not isinstance(beta, numbers.Real)
        or not math.isfinite(float(beta))
        or beta <= 0
    ):
        raise ValueError("beta must be a positive finite scalar")
    if (
        isinstance(gamma, bool)
        or not isinstance(gamma, numbers.Real)
        or not math.isfinite(float(gamma))
        or gamma <= 0
    ):
        raise ValueError("gamma must be a positive finite scalar")
    valid = torch.ones_like(targets, dtype=torch.bool) if ignore_index is None else targets != int(ignore_index)
    invalid = valid & ((targets < 0) | (targets >= vocabulary_size))
    if bool(invalid.any()):
        raise ValueError("targets contain a token ID outside the vocabulary range")
    if bool((~valid).all(dim=-1).any()):
        raise ValueError("each response must contain a non-empty sequence after ignore_index masking")
    safe_targets = targets.masked_fill(~valid, 0).long()
    safe_logits = logits_or_prediction.masked_fill(~valid.unsqueeze(-1), 0)
    token_log_probabilities = torch.log_softmax(safe_logits, dim=-1).gather(
        -1, safe_targets.unsqueeze(-1)
    ).squeeze(-1)
    sequence_lengths = valid.sum(dim=-1)
    sequence_log_probabilities = token_log_probabilities.masked_fill(~valid, 0).sum(dim=-1)
    sequence_log_probabilities = sequence_log_probabilities / sequence_lengths
    reward_difference = float(beta) * (
        sequence_log_probabilities[:, 0] - sequence_log_probabilities[:, 1]
    ) - float(gamma)
    per_pair = torch.nn.functional.softplus(-reward_difference)
    if reduction == "none":
        return per_pair
    if reduction == "sum":
        return per_pair.sum()
    return per_pair.mean()
