import torch
from torch import nn

class DiagonalSSMModel(nn.Module):
    architecture_id = 'diagonal-ssm-model'

    def __init__(self, input_dim=1, width=32, state_size=16, output_dim=1, dt=1.0, task='per_step', dropout=0.0):
        super().__init__()
        if min(input_dim, width, state_size, output_dim) <= 0 or dt <= 0 or task not in ('per_step', 'last') or (not 0 <= dropout < 1):
            raise ValueError('invalid SSM configuration')
        self.input_dim, self.width, self.state_size, self.output_dim, self.dt, self.task = (input_dim, width, state_size, output_dim, dt, task)
        self.input = nn.Linear(input_dim, width)
        self.rate = nn.Parameter(torch.zeros(width, state_size))
        self.drive = nn.Parameter(torch.randn(width, state_size) * 0.02)
        self.read = nn.Parameter(torch.randn(width, state_size) * 0.02)
        self.skip = nn.Linear(width, width, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(width, output_dim)

    def forward(self, x, state=None, return_state=False):
        if x.ndim != 3 or x.shape[-1] != self.input_dim or (not x.is_floating_point()):
            raise ValueError('x must be floating [B,T,input_dim]')
        if x.shape[1] < 1:
            raise ValueError('sequence cannot be empty')
        u = self.input(x)
        b, t, _ = u.shape
        if state is None:
            state = u.new_zeros((b, self.width, self.state_size))
        if state.shape != (b, self.width, self.state_size):
            raise ValueError('state has invalid shape')
        decay = torch.exp(-torch.nn.functional.softplus(self.rate) * self.dt)
        ys = []
        for i in range(t):
            state = state * decay.unsqueeze(0) + u[:, i, :, None] * self.drive.unsqueeze(0)
            ys.append(self.skip(u[:, i]) + torch.sum(state * self.read.unsqueeze(0), dim=-1))
        features = torch.stack(ys, dim=1)
        y = self.head(self.dropout(features[:, -1] if self.task == 'last' else features))
        return (y, state) if return_state else y
