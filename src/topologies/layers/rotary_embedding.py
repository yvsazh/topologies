import torch
from torch import nn

class RotaryEmbedding(nn.Module):
    component_id = 'rotary-embedding'

    def __init__(self, dim, base=10000.0):
        super().__init__()
        if dim <= 0 or base <= 0:
            raise ValueError('dim and base must be positive')
        self.dim = dim
        self.base = base

    def forward(self, x, positions=None):
        d = self.dim - self.dim % 2
        pos = torch.arange(x.shape[-2], device=x.device) if positions is None else torch.as_tensor(positions, device=x.device)
        inv = self.base ** (-torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / max(d, 1))
        a = pos.float()[..., None] * inv
        c = a.cos().to(x.dtype)
        s = a.sin().to(x.dtype)
        z = x[..., :d].reshape(*x.shape[:-1], d // 2, 2)
        u, v = z.unbind(-1)
        r = torch.stack((u * c - v * s, u * s + v * c), -1).flatten(-2)
        return torch.cat((r, x[..., d:]), -1)
