import torch
from torch import nn


class LFQ(nn.Module):
    architecture_id = "lfq"
    def __init__(self, dimensions=4):
        super().__init__()
        if not isinstance(dimensions, int) or isinstance(dimensions, bool) or not 1 <= dimensions <= 63:
            raise ValueError("dimensions must be an integer between 1 and 63")
        self.dimensions = dimensions

    def forward(self, inputs):
        if inputs.shape[-1] != self.dimensions:
            raise ValueError("the final input dimension must match dimensions")
        if not inputs.is_floating_point():
            raise TypeError("inputs must be floating-point latent values")
        signs = torch.where(inputs > 0, 1, -1).to(inputs.dtype)
        quantized = inputs + (signs - inputs).detach()
        bits = (signs > 0).to(torch.int64)
        code = (bits * (2 ** torch.arange(self.dimensions, device=inputs.device))).sum(dim=-1)
        return quantized, code
