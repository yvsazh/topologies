import torch
from torch import nn

class FeedForward(nn.Module):
    component_id = 'feed-forward'

    def __init__(self, d_model, hidden_dim, activation='gelu', dropout=0.0, bias=True):
        super().__init__()
        if min(d_model, hidden_dim) <= 0 or not 0 <= dropout < 1:
            raise ValueError('invalid dimensions/dropout')
        acts = {'relu': nn.ReLU, 'gelu': nn.GELU, 'silu': nn.SiLU}
        if activation in acts:
            self.net = nn.Sequential(nn.Linear(d_model, hidden_dim, bias=bias), acts[activation](), nn.Dropout(dropout), nn.Linear(hidden_dim, d_model, bias=bias), nn.Dropout(dropout))
        elif activation in ('swiglu', 'geglu'):
            self.value = nn.Linear(d_model, hidden_dim, bias=bias)
            self.gate = nn.Linear(d_model, hidden_dim, bias=bias)
            self.out = nn.Linear(hidden_dim, d_model, bias=bias)
            self.kind = activation
            self.drop = nn.Dropout(dropout)
        else:
            raise ValueError('unsupported activation')

    def forward(self, x):
        if hasattr(self, 'net'):
            return self.net(x)
        g = torch.nn.functional.silu(self.gate(x)) if self.kind == 'swiglu' else torch.nn.functional.gelu(self.gate(x))
        return self.drop(self.out(self.drop(self.value(x) * g)))
