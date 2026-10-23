import torch
from torch import nn


class FedAvg(nn.Module):
    component_id = "fedavg"

    def forward(self, client_states, sample_counts):
        if not isinstance(client_states, torch.Tensor):
            try:
                client_states = list(client_states)
            except TypeError:
                raise TypeError("client_states must be a tensor or an iterable of tensors") from None
            if not client_states:
                raise ValueError("client_states must be nonempty")
            if any(not isinstance(state, torch.Tensor) or state.ndim != 1 for state in client_states):
                raise ValueError("each client state must be a one-dimensional tensor")
            if any(state.shape != client_states[0].shape for state in client_states):
                raise ValueError("all client state vectors must have the same length")
            try:
                client_states = torch.stack(client_states)
            except (TypeError, RuntimeError):
                raise ValueError("client state tensors must have compatible dtype and device") from None
        if client_states.ndim not in (2, 3):
            raise ValueError("client_states must be a two-dimensional or batched three-dimensional tensor")
        if any(size == 0 for size in client_states.shape):
            raise ValueError("client_states must be nonempty")
        if not client_states.is_floating_point():
            raise ValueError("client_states must have a floating-point dtype")
        try:
            counts = torch.as_tensor(sample_counts, device=client_states.device)
        except (TypeError, ValueError, RuntimeError):
            raise ValueError("sample_counts must be a one-dimensional sequence of positive integers") from None
        if tuple(counts.shape) != tuple(client_states.shape[:-1]):
            raise ValueError("sample_counts must contain one count per client and batch")
        if counts.dtype == torch.bool or counts.is_floating_point() or counts.is_complex():
            raise ValueError("sample_counts must contain positive integers")
        if torch.any(counts <= 0).item():
            raise ValueError("sample_counts must contain positive integers")
        compute_dtype = torch.float64 if client_states.dtype == torch.float64 else torch.float32
        weights = counts.to(dtype=compute_dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        states = client_states.to(dtype=compute_dtype)
        return (states * weights.unsqueeze(-1)).sum(dim=-2).to(dtype=client_states.dtype)
