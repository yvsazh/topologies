import torch
from torch import nn


class SNIPScore(nn.Module):
    component_id = "snip"
    def __init__(self):
        super().__init__()

    def forward(self, weight, gradient):
        if weight.shape != gradient.shape:
            raise ValueError("weight and gradient must have the same shape")
        return torch.abs(weight * gradient)
