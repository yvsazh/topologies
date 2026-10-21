import torch
from torch import nn


class GEGLU(nn.Module):


    component_id = "geglu"

    def forward(self, x):
        x = torch.as_tensor(x)
        if x.ndim == 0:
            raise ValueError("input must have a last dimension")
        if x.shape[-1] % 2:
            raise ValueError("last dimension must be even")
        value, gate = x.chunk(2, dim=-1)
        return value * torch.nn.functional.gelu(gate)
