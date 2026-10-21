import math
import torch
from torch import nn

class SinusoidalPosition(nn.Module):
    component_id = 'sinusoidal-position'

    def __init__(self, dim):
        super().__init__()
        if dim <= 0:
            raise ValueError('dim must be positive')
        self.dim = dim

    def forward(self, x, positions=None):
        n = x.shape[-2]
        pos = torch.arange(n, device=x.device) if positions is None else torch.as_tensor(positions, device=x.device)
        inv = torch.exp(torch.arange(0, self.dim, 2, device=x.device, dtype=torch.float32) * (-math.log(10000.0) / self.dim))
        a = pos.float().unsqueeze(-1) * inv
        pe = x.new_zeros((len(pos), self.dim))
        pe[..., 0::2] = a.sin().to(x.dtype)
        pe[..., 1::2] = a[..., :pe[..., 1::2].shape[-1]].cos().to(x.dtype)
        return x + pe
