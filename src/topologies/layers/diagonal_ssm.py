import torch
from torch import nn

class DiagonalSSM(nn.Module):
    component_id = 'diagonal-ssm'

    def __init__(self, features, state_size, dt=0.01):
        super().__init__()
        if min(features, state_size) <= 0 or dt <= 0:
            raise ValueError('invalid state-space geometry')
        self.features = features
        self.state_size = state_size
        self.dt = dt
        self.log_decay = nn.Parameter(torch.zeros(features, state_size))
        self.input = nn.Parameter(torch.randn(features, state_size) * 0.02)
        self.output = nn.Parameter(torch.randn(features, state_size) * 0.02)
        self.skip = nn.Parameter(torch.ones(features))

    def step(self, x, state):
        a = torch.exp(-self.log_decay.exp() * self.dt)
        b = (1 - a) * self.input
        state = a * state + x.unsqueeze(-1) * b
        return ((state * self.output).sum(-1) + self.skip * x, state)

    def forward(self, x, state=None):
        if x.ndim != 3 or x.shape[-1] != self.features:
            raise ValueError('expected [B,T,features]')
        state = x.new_zeros((x.shape[0], self.features, self.state_size)) if state is None else state
        ys = []
        for xt in x.unbind(1):
            y, state = self.step(xt, state)
            ys.append(y)
        return (torch.stack(ys, 1), state)
