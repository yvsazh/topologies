import math

import torch
from torch import nn
from torch.nn import functional as F


def _value_error(message):
    return ValueError(message)


class DPO(nn.Module):
    component_id = "dpo"

    def __init__(self, beta=0.1):
        super().__init__()
        if isinstance(beta, bool) or not isinstance(beta, (int, float)):
            raise _value_error("beta must be a positive finite scalar")
        try:
            beta_value = float(beta)
        except (OverflowError, ValueError):
            raise ValueError("beta must be a positive finite scalar") from None
        if not math.isfinite(beta_value) or beta_value <= 0:
            raise ValueError("beta must be a positive finite scalar")
        self.beta = beta_value

    def forward(self, pi_logps, ref_logps, yw_idxs, yl_idxs):
        if not isinstance(pi_logps, torch.Tensor) or not isinstance(ref_logps, torch.Tensor):
            raise _value_error("policy and reference log probabilities must be tensors")
        if pi_logps.shape != ref_logps.shape:
            raise ValueError("policy and reference log probabilities must have the same shape")
        if pi_logps.ndim < 1 or pi_logps.shape[-1] == 0:
            raise ValueError("log probabilities must have a non-empty candidate dimension")
        if not pi_logps.is_floating_point() or not ref_logps.is_floating_point():
            raise ValueError("policy and reference log probabilities must be floating point")
        if pi_logps.device != ref_logps.device:
            raise ValueError("policy and reference log probabilities must be on the same device")
        if not isinstance(yw_idxs, torch.Tensor) or not isinstance(yl_idxs, torch.Tensor):
            raise _value_error("preference indices must be tensors")
        if yw_idxs.shape != yl_idxs.shape:
            raise ValueError("preferred and dispreferred indices must have matching shape")
        if yw_idxs.ndim != pi_logps.ndim:
            raise ValueError("preference indices must match the log probability rank")
        if yw_idxs.shape[:-1] != pi_logps.shape[:-1]:
            raise ValueError("preference index leading dimensions must match log probabilities")
        if yw_idxs.dtype != torch.long or yl_idxs.dtype != torch.long:
            raise ValueError("preference indices must use torch.long dtype")
        if yw_idxs.device != pi_logps.device or yl_idxs.device != pi_logps.device:
            raise ValueError("preference indices must be on the log probability device")
        candidates = pi_logps.shape[-1]
        if bool(((yw_idxs < 0) | (yw_idxs >= candidates)).any()) or bool(
            ((yl_idxs < 0) | (yl_idxs >= candidates)).any()
        ):
            raise ValueError("preference indices are outside the candidate range")

        pi_chosen = pi_logps.gather(-1, yw_idxs)
        pi_rejected = pi_logps.gather(-1, yl_idxs)
        ref_chosen = ref_logps.gather(-1, yw_idxs)
        ref_rejected = ref_logps.gather(-1, yl_idxs)
        logits = self.beta * ((pi_chosen - pi_rejected) - (ref_chosen - ref_rejected))
        return F.softplus(-logits)
