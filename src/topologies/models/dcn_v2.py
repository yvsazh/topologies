import torch
from torch import nn


class DCNV2(nn.Module):
    architecture_id = "dcn-v2"
    def __init__(self, input_dim, deep_dims=(64, 32), num_cross_layers=3, low_rank=16, dropout=0.0):
        super().__init__()
        if input_dim <= 0 or num_cross_layers < 0 or low_rank <= 0:
            raise ValueError("input_dim and low_rank must be positive and num_cross_layers nonnegative")
        self.input_dim = input_dim
        self.cross_u = nn.ParameterList([nn.Parameter(torch.empty(input_dim, low_rank)) for _ in range(num_cross_layers)])
        self.cross_v = nn.ParameterList([nn.Parameter(torch.empty(low_rank, input_dim)) for _ in range(num_cross_layers)])
        self.cross_bias = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(num_cross_layers)])
        for u, v in zip(self.cross_u, self.cross_v):
            nn.init.xavier_uniform_(u)
            nn.init.xavier_uniform_(v)
        layers = []
        width = input_dim
        for dim in deep_dims:
            layers.extend((nn.Linear(width, dim), nn.ReLU()))
            if dropout:
                layers.append(nn.Dropout(dropout))
            width = dim
        self.deep = nn.Sequential(*layers) if layers else nn.Identity()
        self.output = nn.Linear(input_dim + width, 1)

    def cross_forward(self, x):
        x0 = x
        current = x
        for u, v, bias in zip(self.cross_u, self.cross_v, self.cross_bias):
            current = x0 * ((current @ u) @ v + bias) + current
        return current

    def forward(self, x):
        if x.ndim < 2 or x.shape[-1] != self.input_dim:
            raise ValueError("input must have shape (..., input_dim)")
        crossed = self.cross_forward(x)
        deep = self.deep(x)
        return self.output(torch.cat((crossed, deep), dim=-1))
