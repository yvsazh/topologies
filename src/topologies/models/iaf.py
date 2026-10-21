import torch
from torch import nn
from torch.nn import functional as F


class _MaskedLinear(nn.Linear):
    def __init__(self, input_features, output_features, mask):
        super().__init__(input_features, output_features)
        self.register_buffer("mask", mask)

    def forward(self, values):
        return F.linear(values, self.weight * self.mask, self.bias)


class IAF(nn.Module):
    architecture_id = "iaf"
    def __init__(self, dimensions=4, hidden_features=16):
        super().__init__()
        if dimensions < 1 or hidden_features < 1:
            raise ValueError("dimensions and hidden_features must be positive")
        self.dimensions = dimensions
        input_degrees = torch.arange(dimensions)
        hidden_degrees = torch.arange(hidden_features) % dimensions
        output_degrees = torch.arange(dimensions).repeat(2)
        input_mask = (input_degrees.unsqueeze(0) <= hidden_degrees.unsqueeze(1)).to(torch.float32)
        output_mask = (hidden_degrees.unsqueeze(0) < output_degrees.unsqueeze(1)).to(torch.float32)
        self.conditioner = nn.Sequential(
            _MaskedLinear(dimensions, hidden_features, input_mask),
            nn.Tanh(),
            _MaskedLinear(hidden_features, dimensions * 2, output_mask),
        )

    def _parameters_for(self, values):
        shifts, raw_scales = self.conditioner(values).chunk(2, dim=-1)
        return shifts, torch.tanh(raw_scales)

    def forward(self, values):
        if values.shape[-1] != self.dimensions:
            raise ValueError("last input dimension must match dimensions")
        shift, log_scale = self._parameters_for(values)
        return shift + torch.exp(log_scale) * values, log_scale.sum(dim=-1)

    def inverse(self, transformed):
        if transformed.shape[-1] != self.dimensions:
            raise ValueError("last input dimension must match dimensions")
        recovered = torch.zeros_like(transformed)
        for index in range(self.dimensions):
            shift, log_scale = self._parameters_for(recovered)
            value = (transformed[..., index] - shift[..., index]) * torch.exp(-log_scale[..., index])
            recovered = recovered.clone()
            recovered[..., index] = value
        return recovered

    def sample(self, batch_size, device=None, dtype=None):
        parameter = next(self.parameters())
        base = torch.randn(batch_size, self.dimensions, device=device or parameter.device, dtype=dtype or parameter.dtype)
        return self.forward(base)[0]
