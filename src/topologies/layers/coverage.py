import torch
from torch import nn


class CoveragePenalty(nn.Module):
    component_id = "coverage"

    def forward(self, attention, coverage):
        if attention.shape != coverage.shape or attention.ndim < 1:
            raise ValueError("attention and coverage must have the same nonempty shape")
        return torch.minimum(attention, coverage).sum(dim=-1)
