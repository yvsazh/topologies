import torch
from torch import nn

class TimeConditionedResidual(nn.Module):
    component_id = 'time-conditioned-residual'

    def __init__(self, in_channels, out_channels, time_dim, groups=8, condition_dim=None):
        super().__init__()
        if min(in_channels, out_channels, time_dim, groups) <= 0 or in_channels % groups or out_channels % groups:
            raise ValueError('groups must divide channel counts')
        self.n1 = nn.GroupNorm(groups, in_channels)
        self.c1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.n2 = nn.GroupNorm(groups, out_channels)
        self.c2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time = nn.Linear(time_dim, out_channels * 2)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.condition = nn.Conv2d(condition_dim, out_channels, 1) if condition_dim else None

    def forward(self, x, time, conditioning=None):
        y = self.c1(torch.nn.functional.silu(self.n1(x)))
        scale, shift = self.time(time).chunk(2, -1)
        y = self.n2(y) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        if self.condition is not None:
            if conditioning is None:
                raise ValueError('conditioning required')
            y = y + self.condition(conditioning)
        return self.skip(x) + self.c2(torch.nn.functional.silu(y))
