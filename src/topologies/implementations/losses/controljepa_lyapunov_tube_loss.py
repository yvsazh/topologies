import math
import numbers

import torch
import torch.nn.functional as F


def controljepa_lyapunov_tube_loss(
    logits_or_prediction,
    targets,
    *,
    ignore_index=None,
    reduction="mean",
    **params,
):
    if ignore_index is not None:
        raise ValueError("ignore_index must be None for span-index targets")
    if not isinstance(reduction, str) or reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")

    gamma = params.pop("gamma", 0.95)
    tau = params.pop("tau", 1e-4)
    use_softplus = params.pop("use_softplus", True)
    eps = params.pop("eps", 1e-8)
    norm_mode = params.pop("norm_mode", "v_geo")
    anchor_eps = params.pop("anchor_eps", 1e-3)
    if params:
        raise ValueError(f"unsupported parameters: {sorted(params)}")
    if (
        isinstance(gamma, bool)
        or not isinstance(gamma, numbers.Real)
        or not math.isfinite(float(gamma))
        or not 0.0 < float(gamma) < 1.0
    ):
        raise ValueError("gamma must be a finite real number in (0, 1)")
    if (
        isinstance(tau, bool)
        or not isinstance(tau, numbers.Real)
        or not math.isfinite(float(tau))
        or float(tau) < 0.0
    ):
        raise ValueError("tau must be a finite nonnegative real number")
    if use_softplus is not True and use_softplus is not False:
        raise ValueError("use_softplus must be a boolean")
    if (
        isinstance(eps, bool)
        or not isinstance(eps, numbers.Real)
        or not math.isfinite(float(eps))
        or float(eps) <= 0.0
    ):
        raise ValueError("eps must be a finite positive real number")
    if norm_mode not in ("v_geo", "d_model", "d_t"):
        raise ValueError("norm_mode must be 'v_geo', 'd_model', or 'd_t'")
    if (
        isinstance(anchor_eps, bool)
        or not isinstance(anchor_eps, numbers.Real)
        or not math.isfinite(float(anchor_eps))
        or float(anchor_eps) < 0.0
    ):
        raise ValueError("anchor_eps must be a finite nonnegative real number")

    h = logits_or_prediction
    if not isinstance(h, torch.Tensor) or not h.is_floating_point():
        raise ValueError("logits_or_prediction must be a floating-point tensor")
    if h.ndim != 3 or min(h.shape) < 1:
        raise ValueError("logits_or_prediction must have nonempty shape [B, T, D]")
    if not isinstance(targets, torch.Tensor) or targets.ndim != 2:
        raise ValueError("targets must be an integral tensor with shape [B, 4]")
    if targets.shape != (h.shape[0], 4):
        raise ValueError("targets must have shape [B, 4] matching the batch")
    if targets.is_floating_point() or targets.is_complex() or targets.dtype == torch.bool:
        raise ValueError("targets must contain integral span indices")

    batch_size, sequence_length, feature_size = h.shape
    bounds = targets.tolist()
    displacements = torch.zeros_like(h)
    reference_vectors = h.new_zeros((batch_size, feature_size))
    valid = torch.zeros(
        (batch_size, sequence_length - 1), dtype=torch.bool, device=h.device
    )

    for batch_index, row in enumerate(bounds):
        user_start, user_end, assistant_start, assistant_end = row
        if not (
            0 <= user_start <= user_end < sequence_length
            and 0 <= assistant_start <= assistant_end < sequence_length
        ):
            continue
        user_chord = h[batch_index, user_end] - h[batch_index, user_start]
        assistant_chord = h[batch_index, assistant_end] - h[batch_index, assistant_start]
        reference_vectors[batch_index] = user_chord + assistant_chord
        displacements[batch_index, user_start : user_end + 1] = (
            h[batch_index, user_start : user_end + 1] - h[batch_index, user_start]
        )
        displacements[batch_index, assistant_start : assistant_end + 1] = (
            user_chord
            + h[batch_index, assistant_start : assistant_end + 1]
            - h[batch_index, assistant_start]
        )
        valid[batch_index, user_start:user_end] = True
        valid[batch_index, assistant_start:assistant_end] = True

    reference_norm = torch.linalg.vector_norm(reference_vectors, dim=-1, keepdim=True)
    reference_direction = reference_vectors / reference_norm.clamp_min(float(eps))
    transversal = displacements - (
        displacements * reference_direction.unsqueeze(1)
    ).sum(dim=-1, keepdim=True) * reference_direction.unsqueeze(1)
    transversal_sq = transversal.square().sum(dim=-1)

    if norm_mode == "v_geo":
        reference_sq = reference_vectors.square().sum(dim=-1).clamp_min(float(eps))
        candidate = transversal_sq / reference_sq.unsqueeze(1)
    elif norm_mode == "d_model":
        candidate = transversal_sq / feature_size
    else:
        displacement_sq = displacements.square().sum(dim=-1)
        reference_sq = reference_vectors.square().sum(dim=-1).clamp_min(float(eps))
        if float(anchor_eps) > 0.0:
            denominator = float(anchor_eps) * reference_sq.unsqueeze(1) + displacement_sq
        else:
            denominator = displacement_sq.clamp_min(float(eps))
        candidate = transversal_sq / denominator

    violation = candidate[:, 1:] - float(gamma) * candidate[:, :-1] - float(tau)
    penalty = F.softplus(violation) if use_softplus else F.relu(violation)
    per_transition = torch.where(valid, penalty, torch.zeros_like(penalty)) + h.sum() * 0.0
    if reduction == "none":
        return per_transition
    if reduction == "sum":
        return per_transition.sum()
    valid_count = valid.sum().clamp_min(1).to(dtype=h.dtype)
    return per_transition.sum() / valid_count
