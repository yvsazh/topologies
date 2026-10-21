import torch
from torch import nn


class NICE(nn.Module):
    architecture_id = "nice"
    def __init__(self, input_dim=4, hidden_dim=8):
        super().__init__()
        if input_dim < 2 or input_dim % 2:
            raise ValueError("input_dim must be an even integer of at least two")
        self.input_dim = input_dim
        split = input_dim // 2
        self.translation = nn.Sequential(
            nn.Linear(split, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, split),
        )

    def forward(self, x, inverse=False):
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"expected final dimension {self.input_dim}")
        first, second = x.chunk(2, dim=-1)
        shift = self.translation(first)
        transformed = second - shift if inverse else second + shift
        return torch.cat((first, transformed), dim=-1)

    def inverse(self, y):
        return self.forward(y, inverse=True)

    def log_abs_det_jacobian(self, x=None):
        if x is None:
            return torch.tensor(0.0)
        return x.new_zeros(x.shape[:-1])
