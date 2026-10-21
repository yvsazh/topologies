import torch
from torch import nn
import torch.nn.functional as F

class _Block(nn.Module):

    def __init__(self, cin, cout, kernel, dilation, dropout):
        super().__init__()
        self.kernel, self.dilation = (kernel, dilation)
        self.conv1 = nn.Conv1d(cin, cout, kernel, dilation=dilation)
        self.conv2 = nn.Conv1d(cout, cout, kernel, dilation=dilation)
        self.skip = nn.Identity() if cin == cout else nn.Conv1d(cin, cout, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        padding = (self.kernel - 1) * self.dilation
        y = self.conv1(F.pad(x, (padding, 0))).relu()
        y = self.drop(y)
        y = self.conv2(F.pad(y, (padding, 0)))
        return (self.skip(x) + y).relu()

class TemporalConvNet(nn.Module):
    architecture_id = 'temporal-convnet'

    def __init__(self, in_features=1, out_features=1, channels=(16, 16, 16), kernel_size=3, dropout=0.0, head='per_step', horizon=1):
        super().__init__()
        channels = tuple(channels)
        if in_features < 1 or out_features < 1 or (not channels) or (min(channels) < 1) or (kernel_size < 1) or (horizon < 1):
            raise ValueError('dimensions and kernel_size must be positive')
        if head not in ('per_step', 'last', 'horizon') or not 0 <= dropout < 1:
            raise ValueError('invalid head or dropout')
        self.head_kind, self.horizon = (head, horizon)
        self.in_features = in_features
        blocks = []
        width = in_features
        for i, next_width in enumerate(channels):
            blocks.append(_Block(width, next_width, kernel_size, 2 ** i, dropout))
            width = next_width
        self.network = nn.Sequential(*blocks)
        self.projection = nn.Linear(width, out_features * (horizon if head == 'horizon' else 1))
        self.out_features = out_features

    def forward(self, x):
        if x.ndim != 3 or not x.is_floating_point() or x.shape[-1] != self.in_features:
            raise ValueError('x must be floating [B,T,in_features]')
        if x.shape[1] < 1:
            raise ValueError('sequence cannot be empty')
        y = self.network(x.transpose(1, 2)).transpose(1, 2)
        if self.head_kind == 'last':
            y = y[:, -1]
        elif self.head_kind == 'horizon':
            return self.projection(y[:, -1]).reshape(x.shape[0], self.horizon, self.out_features)
        return self.projection(y)
