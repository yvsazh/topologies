import torch
from torch import nn
import torch.nn.functional as F

class CausalTemporalBlock(nn.Module):
    component_id = 'causal-temporal-block'

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, gated=False):
        super().__init__()
        if min(in_channels, out_channels, kernel_size, dilation) <= 0:
            raise ValueError('dimensions must be positive')
        self.left = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels * (2 if gated else 1), kernel_size, dilation=dilation)
        self.gated = gated
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        y = self.conv(F.pad(x, (self.left, 0)))
        if self.gated:
            a, b = y.chunk(2, 1)
            y = torch.tanh(a) * torch.sigmoid(b)
        return F.relu(y + self.skip(x))
