import torch
from torch import nn


class PPGAuxiliaryValueLoss(nn.Module):
    architecture_id = "ppg"
    PRESETS = {"tiny": {}, "small": {}}

    def __init__(self, *args, **kwargs):
        if args or kwargs:
            raise ValueError("PPGAuxiliaryValueLoss accepts no constructor arguments")
        super().__init__()

    def forward(self, auxiliary_values, value_targets):
        if not torch.is_tensor(auxiliary_values) or not torch.is_tensor(value_targets):
            raise ValueError("auxiliary_values and value_targets must be tensors")
        if auxiliary_values.shape != value_targets.shape:
            raise ValueError("auxiliary_values and value_targets must have matching shapes")
        if auxiliary_values.ndim == 0:
            raise ValueError("auxiliary_values and value_targets must have non-scalar shapes")
        if auxiliary_values.numel() == 0:
            raise ValueError("auxiliary_values and value_targets must be nonempty")
        if not torch.is_floating_point(auxiliary_values) or not torch.is_floating_point(
            value_targets
        ):
            raise ValueError("auxiliary_values and value_targets must be floating-point tensors")
        if auxiliary_values.dtype != value_targets.dtype:
            raise ValueError("auxiliary_values and value_targets must have matching dtypes")
        if auxiliary_values.device != value_targets.device:
            raise ValueError("auxiliary_values and value_targets must be on the same device")
        return 0.5 * (auxiliary_values - value_targets).square().mean()
