import torch
from torch import nn


class FedNova(nn.Module):
    component_id = "fednova"

    def __init__(self):
        super().__init__()

    def forward(self, client_updates, sample_counts, steps):
        if not isinstance(client_updates, torch.Tensor):
            raise TypeError("client_updates must be a tensor")
        if client_updates.ndim not in (2, 3):
            raise ValueError("client_updates must be two-dimensional or batched three-dimensional")
        if any(size == 0 for size in client_updates.shape):
            raise ValueError("client_updates must have nonempty client and parameter dimensions")
        if not client_updates.is_floating_point():
            raise ValueError("client_updates must have a floating-point dtype")

        expected_metadata_shape = client_updates.shape[:-1]
        if not isinstance(sample_counts, torch.Tensor):
            raise TypeError("sample_counts must be a tensor")
        if tuple(sample_counts.shape) != tuple(expected_metadata_shape):
            raise ValueError("sample_counts must contain one count per client and batch")
        if sample_counts.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ) or torch.any(sample_counts <= 0):
            raise ValueError("sample_counts must contain positive integers")

        if not isinstance(steps, torch.Tensor):
            raise TypeError("steps must be a tensor")
        if tuple(steps.shape) != tuple(expected_metadata_shape):
            raise ValueError("steps must contain one step count per client and batch")
        if steps.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ) or torch.any(steps <= 0):
            raise ValueError("steps must contain positive integers")

        counts = sample_counts.to(device=client_updates.device, dtype=client_updates.dtype)
        local_steps = steps.to(device=client_updates.device, dtype=client_updates.dtype)
        weights = counts / counts.sum(dim=-1, keepdim=True)
        effective_steps = torch.sum(weights * local_steps, dim=-1)
        normalized_updates = client_updates / local_steps.unsqueeze(-1)
        aggregated_updates = torch.sum(weights.unsqueeze(-1) * normalized_updates, dim=-2)
        return effective_steps.unsqueeze(-1) * aggregated_updates
