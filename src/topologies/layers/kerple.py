import torch
from torch import nn
from torch.nn import functional as F


class KERPLE(nn.Module):
    component_id = "kerple"
    def __init__(self, num_heads=1):
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        self.num_heads = num_heads
        initial_raw = torch.log(torch.expm1(torch.ones(num_heads)))
        self.raw_r1 = nn.Parameter(initial_raw.clone())
        self.raw_r2 = nn.Parameter(initial_raw.clone())

    @property
    def r1(self):
        return F.softplus(self.raw_r1)

    @property
    def r2(self):
        return F.softplus(self.raw_r2)

    def forward(self, query_positions, key_positions=None):
        if key_positions is None:
            key_positions = query_positions
        query_positions = torch.as_tensor(query_positions)
        key_positions = torch.as_tensor(key_positions, device=query_positions.device)
        offset = (query_positions[..., :, None] - key_positions[..., None, :]).abs()
        r1 = self.r1.view(*([1] * (offset.ndim - 2)), self.num_heads, 1, 1)
        r2 = self.r2.view(*([1] * (offset.ndim - 2)), self.num_heads, 1, 1)
        offset = offset.unsqueeze(-3)
        return -r1 * torch.log1p(r2 * offset)
