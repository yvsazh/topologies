import math
import numbers

import torch
import torch.nn.functional as F


def _edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_token in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_token in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hypothesis_index] + 1,
                    previous[hypothesis_index - 1]
                    + (reference_token != hypothesis_token),
                )
            )
        previous = current
    return previous[-1]


def _word_sequence(sequence, ignore_index, eos_token_id):
    words = []
    for token in sequence:
        if ignore_index is not None and token == ignore_index:
            continue
        if eos_token_id is not None and token == eos_token_id:
            break
        words.append(token)
    return words


def _valid_ce_weight(value):
    if not isinstance(value, numbers.Real) or isinstance(value, bool):
        return False
    try:
        value = float(value)
    except (OverflowError, ValueError):
        return False
    return math.isfinite(value) and value >= 0


def minimum_wer(
    logits_or_prediction,
    targets,
    *,
    ignore_index=None,
    reduction="mean",
    **params,
):
    unknown_params = set(params) - {"ce_weight", "eos_token_id", "nbest_size"}
    if unknown_params:
        raise ValueError(f"unsupported parameters: {sorted(unknown_params)}")
    ce_weight = params.get("ce_weight", 0.01)
    eos_token_id = params.get("eos_token_id")
    nbest_size = params.get("nbest_size")
    if not isinstance(logits_or_prediction, torch.Tensor) or not logits_or_prediction.is_floating_point():
        raise ValueError("logits_or_prediction must be a floating-point tensor")
    if logits_or_prediction.ndim != 4:
        raise ValueError("logits_or_prediction must have shape [B, N+1, T, V]")
    batch_size, rows, time_steps, vocab_size = logits_or_prediction.shape
    if batch_size < 1 or rows < 2 or time_steps < 1 or vocab_size < 1:
        raise ValueError("logits_or_prediction must have nonempty batch, candidate, time, and vocabulary axes")
    if not isinstance(targets, torch.Tensor) or targets.ndim != 3:
        raise ValueError("targets must have shape [B, N+1, T]")
    if tuple(targets.shape) != (batch_size, rows, time_steps):
        raise ValueError("targets shape must match the logits batch, candidate, and time axes")
    if targets.device != logits_or_prediction.device:
        raise ValueError("targets and logits_or_prediction must be on the same device")
    if targets.dtype == torch.bool or targets.is_complex():
        raise ValueError("targets must contain integer token IDs")
    if targets.is_floating_point():
        if not bool(torch.isfinite(targets).all()) or not bool((targets == targets.round()).all()):
            raise ValueError("targets must contain finite integer token IDs")
    elif targets.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise ValueError("targets must contain integer token IDs")
    if ignore_index is not None and (
        isinstance(ignore_index, bool) or not isinstance(ignore_index, numbers.Integral)
    ):
        raise ValueError("ignore_index must be an integer or None")
    if eos_token_id is not None and (
        isinstance(eos_token_id, bool) or not isinstance(eos_token_id, numbers.Integral)
    ):
        raise ValueError("eos_token_id must be an integer or None")
    if eos_token_id is not None and not 0 <= eos_token_id < vocab_size:
        raise ValueError("eos_token_id must be in the vocabulary")
    if eos_token_id is not None and eos_token_id == ignore_index:
        raise ValueError("eos_token_id cannot equal ignore_index")
    if not _valid_ce_weight(ce_weight):
        raise ValueError("ce_weight must be finite and nonnegative")
    if nbest_size is not None and (
        isinstance(nbest_size, bool)
        or not isinstance(nbest_size, numbers.Integral)
        or nbest_size != rows - 1
    ):
        raise ValueError("nbest_size must equal the number of candidate rows")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")

    token_ids = targets.long()
    valid_tokens = torch.ones_like(token_ids, dtype=torch.bool)
    if ignore_index is not None:
        valid_tokens = token_ids != ignore_index
    if bool((valid_tokens & ((token_ids < 0) | (token_ids >= vocab_size))).any()):
        raise ValueError("nonignored target token IDs must be in the vocabulary")
    if eos_token_id is None:
        active_tokens = valid_tokens
    else:
        eos_tokens = token_ids.eq(eos_token_id) & valid_tokens
        eos_tokens_before = eos_tokens.long().cumsum(dim=-1) - eos_tokens.long()
        active_tokens = valid_tokens & eos_tokens_before.eq(0)

    safe_ids = token_ids.masked_fill(~active_tokens, 0)
    log_probabilities = F.log_softmax(logits_or_prediction, dim=-1)
    token_log_probabilities = log_probabilities.gather(
        -1, safe_ids.unsqueeze(-1)
    ).squeeze(-1)
    sequence_scores = (
        token_log_probabilities[:, 1:] * active_tokens[:, 1:].to(logits_or_prediction.dtype)
    ).sum(dim=-1)
    candidate_probabilities = sequence_scores.softmax(dim=1)

    cpu_targets = token_ids.detach().cpu().tolist()
    distances = []
    for batch_index in range(batch_size):
        reference = _word_sequence(
            cpu_targets[batch_index][0], ignore_index, eos_token_id
        )
        distances.append(
            [
                _edit_distance(
                    reference,
                    _word_sequence(
                        cpu_targets[batch_index][candidate_index],
                        ignore_index,
                        eos_token_id,
                    ),
                )
                for candidate_index in range(1, rows)
            ]
        )
    distance_tensor = logits_or_prediction.new_tensor(distances)
    expected_risk = (candidate_probabilities * distance_tensor).sum(dim=1)
    reference_nll = -(
        token_log_probabilities[:, 0]
        * active_tokens[:, 0].to(logits_or_prediction.dtype)
    ).sum(dim=1)
    losses = expected_risk + float(ce_weight) * reference_nll
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.mean()
