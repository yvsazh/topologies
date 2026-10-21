import torch
from torch import nn


class HardSwish(nn.Module):
    component_id = "hard-swish"
    def forward(self, x):
        return x * torch.clamp(x + 3, min=0, max=6) / 6
