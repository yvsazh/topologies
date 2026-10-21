import torch
from torch import nn

class RMSNorm(nn.Module):
    component_id = 'rmsnorm'

    def __init__(self, normalized_shape, eps=1e-06, affine=True):
        super().__init__()
        if isinstance(normalized_shape, bool) or not isinstance(normalized_shape, int) or normalized_shape <= 0 or (eps <= 0):
            raise ValueError('shape must be a positive integer and eps positive')
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape)) if affine else None

    def forward(self, x):
        y = x * (x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps).rsqrt().to(x.dtype)
        return y if self.weight is None else y * self.weight
