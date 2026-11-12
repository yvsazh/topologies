import torch
from torch import nn


class OpenVLA(nn.Module):
    component_id = "openvla"

    def __init__(self, action_dim=7):
        if isinstance(action_dim, bool) or not isinstance(action_dim, int) or action_dim < 1:
            raise ValueError("action_dim must be a positive integer")
        super().__init__()
        self.action_dim = action_dim

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            error = ValueError
            raise error("x must be a torch.Tensor")
        if x.ndim < 1:
            raise ValueError("x must have rank at least 1")
        if x.shape[-1] != self.action_dim:
            raise ValueError(f"action width must be {self.action_dim}")
        return x
