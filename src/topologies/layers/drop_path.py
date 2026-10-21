import torch
from torch import nn

class DropPath(nn.Module):
    component_id = 'drop-path'

    def __init__(self, drop_prob=0.0):
        super().__init__()
        if not 0 <= drop_prob < 1:
            raise ValueError('drop_prob must be in [0,1)')
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep
