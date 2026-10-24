import torch
from torch import nn


class SquaredReLU(nn.Module):
    component_id = "squared-relu"

    def __init__(self, *args, **kwargs):
        if args or kwargs:
            raise ValueError("SquaredReLU does not accept constructor arguments")
        super().__init__()

    def forward(self, x):
        return torch.relu(x).square()
