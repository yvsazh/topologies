import torch
from torch import nn

class FMInteraction(nn.Module):
    component_id = 'fm-interaction'

    def forward(self, fields):
        if fields.ndim < 3:
            raise ValueError('expected [..., fields, embedding]')
        summed = fields.sum(dim=-2)
        return 0.5 * (summed.square() - fields.square().sum(dim=-2)).sum(dim=-1, keepdim=True)
