import torch
from torch import nn


class InceptionResNetResidualScale(nn.Module):
    component_id = "inception-resnet"

    def __init__(self, scale=0.1):
        super().__init__()
        if not 0.0 < scale <= 1.0:
            raise ValueError("scale must be in (0, 1]")
        self.scale = scale

    def forward(self, x, residual):
        if x.shape != residual.shape:
            raise ValueError("residual branch output must match the input shape")
        return x + self.scale * residual
