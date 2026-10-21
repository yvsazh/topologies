import torch
from torch import nn


class ConditionalNeuralProcess(nn.Module):
    component_id = "cnp"
    def __init__(self):
        super().__init__()

    def forward(self, context):
        if context.ndim < 2:
            raise ValueError("context must include a context axis at dimension 1")
        if context.shape[1] == 0:
            raise ValueError("context must not be empty")
        return context.mean(dim=1)
