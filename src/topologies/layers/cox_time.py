import math

import torch
from torch import nn


class CoxTime(nn.Module):
    component_id = "cox-time"

    PRESETS = {"tiny": {"shrink": 0.0}, "small": {"shrink": 0.001}}

    @staticmethod
    def _raise_value_error(message):
        raise ValueError(message)

    def __init__(self, shrink=0.0):
        super().__init__()
        if not isinstance(shrink, (int, float)) or isinstance(shrink, bool):
            self._raise_value_error("shrink must be a finite nonnegative number")
        try:
            shrink_value = float(shrink)
        except (OverflowError, ValueError):
            self._raise_value_error("shrink must be a finite nonnegative number")
        if not math.isfinite(shrink_value) or shrink_value < 0.0:
            self._raise_value_error("shrink must be a finite nonnegative number")
        self.shrink = shrink_value

    def forward(self, event_log_risk, control_log_risk, reduction="mean"):
        if not isinstance(event_log_risk, torch.Tensor) or not isinstance(control_log_risk, torch.Tensor):
            self._raise_value_error("event_log_risk and control_log_risk must be tensors")
        if not isinstance(reduction, str) or reduction not in ("mean", "sum", "none"):
            self._raise_value_error("reduction must be 'mean', 'sum', or 'none'")
        if event_log_risk.ndim == 2 and event_log_risk.shape[1] == 1:
            event_log_risk = event_log_risk.squeeze(1)
        elif event_log_risk.ndim != 1:
            raise ValueError("event_log_risk must be one-dimensional with shape [B]")
        if control_log_risk.ndim != 2:
            raise ValueError("control_log_risk must be two-dimensional with shape [B, C]")
        if event_log_risk.shape[0] == 0:
            raise ValueError("event batch must be non-empty")
        if event_log_risk.shape[0] != control_log_risk.shape[0]:
            raise ValueError("event and control scores must have the same batch size")
        if control_log_risk.shape[1] == 0:
            raise ValueError("each event must have at least one control")
        if not event_log_risk.is_floating_point() or not control_log_risk.is_floating_point():
            raise ValueError("event and control scores must be floating-point tensors")
        if event_log_risk.device != control_log_risk.device or event_log_risk.dtype != control_log_risk.dtype:
            raise ValueError("event and control scores must share a device and dtype")

        differences = control_log_risk - event_log_risk.unsqueeze(1)
        zero_difference = torch.zeros_like(event_log_risk).unsqueeze(1)
        event_losses = torch.logsumexp(torch.cat((zero_difference, differences), dim=1), dim=1)
        event_penalties = self.shrink * (event_log_risk.abs() + control_log_risk.abs().sum(dim=1))
        if reduction == "none":
            return event_losses + event_penalties
        if reduction == "sum":
            return event_losses.sum() + event_penalties.sum()
        return event_losses.mean() + event_penalties.sum()
