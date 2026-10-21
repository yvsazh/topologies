import torch
from torch import nn


class FeatureAnomaly(nn.Module):
    component_id = "efficientad"
    def __init__(self):
        super().__init__()

    def forward(self, student, teacher):
        return (student - teacher).square().mean(dim=1)
