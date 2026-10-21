import torch
from torch import nn


class NeuralUCBScore(nn.Module):
    component_id = "neuralucb-bonus"
    def __init__(self, beta=1.0):
        super().__init__()
        if beta < 0:
            raise ValueError("beta must be nonnegative")
        self.beta = beta

    def forward(self, mean, uncertainty):
        if torch.any(torch.as_tensor(uncertainty) < 0).item():
            raise ValueError("uncertainty must be nonnegative")
        return mean + self.beta * uncertainty
