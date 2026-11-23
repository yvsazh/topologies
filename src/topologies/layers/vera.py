import math

import torch
from torch import nn


class VeRA(nn.Module):
    component_id = "vera"
    PRESETS = {
        "tiny": {"in_features": 4, "out_features": 3, "rank": 1, "d_init": 0.1},
        "small": {"in_features": 32, "out_features": 32, "rank": 4, "d_init": 0.1},
        "base": {"in_features": 128, "out_features": 128, "rank": 8, "d_init": 0.1},
    }

    def __init__(self, in_features, out_features, rank, d_init=0.1, shared_A=None, shared_B=None):
        super().__init__()
        for name, value in (("in_features", in_features), ("out_features", out_features), ("rank", rank)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(d_init, bool) or not isinstance(d_init, (int, float)):
            error = ValueError("d_init must be a finite nonzero number")
            raise error
        try:
            d_init = float(d_init)
        except OverflowError as exc:
            raise ValueError("d_init must be a finite nonzero number") from exc
        if not math.isfinite(d_init) or d_init == 0.0:
            raise ValueError("d_init must be a finite nonzero number")
        if (shared_A is None) != (shared_B is None):
            raise ValueError("shared_A and shared_B must be supplied together")
        if shared_A is None:
            A = torch.empty(rank, in_features)
            B = torch.empty(out_features, rank)
            nn.init.kaiming_uniform_(A)
            nn.init.kaiming_uniform_(B)
        else:
            if not isinstance(shared_A, torch.Tensor) or not isinstance(shared_B, torch.Tensor):
                raise ValueError("shared_A and shared_B must be tensors")
            if not shared_A.is_floating_point() or not shared_B.is_floating_point():
                raise ValueError("shared_A and shared_B must be floating-point tensors")
            if tuple(shared_A.shape) != (rank, in_features):
                raise ValueError(f"shared_A must have shape ({rank}, {in_features})")
            if tuple(shared_B.shape) != (out_features, rank):
                raise ValueError(f"shared_B must have shape ({out_features}, {rank})")
            if shared_A.dtype != shared_B.dtype or shared_A.device != shared_B.device:
                raise ValueError("shared_A and shared_B must have the same dtype and device")
            A = shared_A.detach()
            B = shared_B.detach()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.register_buffer("A", A)
        self.register_buffer("B", B)
        self.b = nn.Parameter(torch.zeros(out_features, dtype=A.dtype, device=A.device))
        self.d = nn.Parameter(torch.full((rank,), d_init, dtype=A.dtype, device=A.device))

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim == 0 or not x.is_floating_point():
            raise ValueError("x must be a floating-point tensor with a final feature dimension")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"x must have a final dimension of {self.in_features}")
        if x.dtype != self.A.dtype or x.device != self.A.device:
            raise ValueError("x must have the same dtype and device as the VeRA matrices")
        return ((x @ self.A.transpose(-2, -1)) * self.d) @ self.B.transpose(-2, -1) * self.b
