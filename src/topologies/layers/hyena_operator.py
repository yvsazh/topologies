import torch
import torch.nn as nn
import torch.nn.functional as F


class HyenaOperator(nn.Module):
    component_id = "hyena-operator"

    def __init__(self, d_model=8, filter_dim=16, dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        self.d_model = d_model
        self.position_filter = nn.Sequential(nn.Linear(1, filter_dim), nn.SiLU(), nn.Linear(filter_dim, d_model))
        self.in_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

    def filter_from_positions(self, positions):
        return self.position_filter(positions.unsqueeze(-1))

    def _causal_convolution(self, values, kernel):
        batch, length, channels = values.shape
        weights = kernel.transpose(0, 1).flip(-1).unsqueeze(1)
        padded = F.pad(values.transpose(1, 2), (length - 1, 0))
        return F.conv1d(padded, weights, groups=channels).transpose(1, 2)

    def forward(self, x, gate=None):
        x = x.to(dtype=self.in_proj.weight.dtype)
        length = x.shape[1]
        positions = torch.arange(length, device=x.device, dtype=x.dtype) / max(length, 1)
        kernel = self.filter_from_positions(positions)
        first, second, third = self.in_proj(x).chunk(3, dim=-1)
        if gate is not None:
            second = second * gate
        mixed = self._causal_convolution(first, kernel) * second
        return self.out_proj(self._causal_convolution(mixed, kernel) * third)
