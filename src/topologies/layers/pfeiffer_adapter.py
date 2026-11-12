import torch
from torch import nn


class PfeifferAdapter(nn.Module):
    component_id = "pfeiffer-adapter"
    PRESETS = {
        "tiny": {"hidden_size": 8, "reduction_factor": 2},
        "small": {"hidden_size": 64, "reduction_factor": 2},
        "base": {"hidden_size": 768, "reduction_factor": 2},
    }

    def __init__(self, hidden_size, reduction_factor=16):
        super().__init__()
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        if (
            isinstance(reduction_factor, bool)
            or not isinstance(reduction_factor, int)
            or reduction_factor <= 0
        ):
            raise ValueError("reduction_factor must be a positive integer")

        bottleneck_size = max(1, hidden_size // reduction_factor)
        self.hidden_size = hidden_size
        self.down_proj = nn.Linear(hidden_size, bottleneck_size)
        self.activation = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck_size, hidden_size)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, hidden, residual):
        if not isinstance(hidden, torch.Tensor) or hidden.ndim == 0:
            raise ValueError("hidden must be a tensor with at least one dimension")
        if hidden.shape[-1] != self.hidden_size:
            raise ValueError(f"hidden last dimension must equal hidden_size ({self.hidden_size})")
        if not isinstance(residual, torch.Tensor) or residual.shape != hidden.shape:
            raise ValueError("residual must be a tensor with the same shape as hidden")
        return self.up_proj(self.activation(self.down_proj(hidden))) + residual
