import torch
from torch import nn


class ANNtoSNN(nn.Module):
    component_id = "ann-snn"

    def __init__(self):
        super().__init__()

    def forward(self, weights, spike_inputs):
        if not isinstance(weights, torch.Tensor):
            raise TypeError("weights must be a torch.Tensor")
        if not isinstance(spike_inputs, torch.Tensor):
            raise TypeError("spike_inputs must be a torch.Tensor")
        if weights.ndim != 2:
            raise ValueError("weights must have shape [out_features, in_features]")
        if spike_inputs.ndim != 3:
            raise ValueError("spike_inputs must have shape [time_steps, batch, in_features]")
        if any(size == 0 for size in weights.shape):
            raise ValueError("weights dimensions must be nonempty")
        if any(size == 0 for size in spike_inputs.shape):
            raise ValueError("spike_inputs dimensions must be nonempty")
        if weights.shape[1] != spike_inputs.shape[2]:
            raise ValueError("weights and spike_inputs feature dimension must match")
        if not weights.is_floating_point() or not spike_inputs.is_floating_point():
            raise ValueError("weights and spike_inputs must have floating-point dtypes")
        if weights.dtype != spike_inputs.dtype:
            raise ValueError("weights and spike_inputs must have the same dtype")
        if weights.device != spike_inputs.device:
            raise ValueError("weights and spike_inputs must be on the same device")
        if not torch.isfinite(weights).all().item():
            raise ValueError("weights must contain only finite values")
        if not torch.isfinite(spike_inputs).all().item():
            raise ValueError("spike_inputs must contain only finite values")
        weighted_currents = torch.matmul(spike_inputs, weights.transpose(0, 1))
        if not torch.isfinite(weighted_currents).all().item():
            raise ValueError("weighted currents must contain only finite values")
        threshold = weighted_currents.amax().clamp_min(0.0)
        if threshold.item() <= 0.0:
            raise ValueError("spike_inputs must produce a positive maximum weighted current")
        return weights / threshold
