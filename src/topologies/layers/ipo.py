import math
from numbers import Real

import torch
from torch import nn


class IPO(nn.Module):
    component_id = "ipo"

    def __init__(self, tau: float):
        super().__init__()
        if isinstance(tau, bool) or not isinstance(tau, Real) or not math.isfinite(tau) or tau <= 0:
            raise ValueError("tau must be a positive finite number")
        self.tau = float(tau)

    def forward(
        self,
        chosen_logp: torch.Tensor,
        rejected_logp: torch.Tensor,
        chosen_ref_logp: torch.Tensor,
        rejected_ref_logp: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        if reduction not in ("mean", "none"):
            raise ValueError("reduction must be 'mean' or 'none'")
        values = (chosen_logp, rejected_logp, chosen_ref_logp, rejected_ref_logp)
        if any(not isinstance(value, torch.Tensor) for value in values):
            raise ValueError("all log probability inputs must be tensors")
        if any(value.ndim not in (1, 2) or value.shape[0] == 0
               or (value.ndim == 2 and value.shape[1] != 1) for value in values):
            raise ValueError("log probability inputs must have non-empty shape [B] or [B, 1]")
        values = tuple(value.squeeze(-1) if value.ndim == 2 else value for value in values)
        if any(value.shape != values[0].shape for value in values[1:]):
            raise ValueError("all log probability inputs must have the same shape")
        if any(not value.is_floating_point() for value in values):
            raise ValueError("log probability inputs must have floating point dtype")
        chosen_logp, rejected_logp, chosen_ref_logp, rejected_ref_logp = values
        if any(value.dtype != chosen_logp.dtype or value.device != chosen_logp.device
               for value in values[1:]):
            raise ValueError("log probability inputs must share dtype and device")
        gap = (chosen_logp - chosen_ref_logp) - (rejected_logp - rejected_ref_logp)
        losses = (gap - 1.0 / (2.0 * self.tau)).square()
        return losses.mean() if reduction == "mean" else losses
