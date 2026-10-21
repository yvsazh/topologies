import torch
from torch import nn

class ConvNormAct(nn.Module):
    component_id = 'conv-norm-act'

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1, dimension=2, norm=None, activation='relu', dropout=0.0, bias=None):
        super().__init__()
        if min(in_channels, out_channels, kernel_size, stride, dilation, groups, dimension) <= 0 or dimension not in (1, 2) or in_channels % groups or out_channels % groups or (not 0 <= dropout < 1):
            raise ValueError('invalid convolution geometry or dropout')
        conv = nn.Conv1d if dimension == 1 else nn.Conv2d
        self.conv = conv(in_channels, out_channels, kernel_size, stride=stride, padding=dilation * (kernel_size // 2), dilation=dilation, groups=groups, bias=norm is None if bias is None else bias)
        if norm not in (None, 'batch', 'group'):
            raise ValueError('norm must be None, batch, or group')
        group_count = min(8, out_channels)
        while out_channels % group_count:
            group_count -= 1
        self.norm = nn.Identity() if norm is None else (nn.BatchNorm1d if dimension == 1 else nn.BatchNorm2d)(out_channels) if norm == 'batch' else nn.GroupNorm(group_count, out_channels)
        acts = {'relu': nn.ReLU, 'gelu': nn.GELU, 'silu': nn.SiLU, 'identity': nn.Identity}
        if activation not in acts:
            raise ValueError('unsupported activation')
        self.activation = acts[activation]()
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x):
        return self.dropout(self.activation(self.norm(self.conv(x))))
