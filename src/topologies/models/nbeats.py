import math
import torch
from torch import nn

class _Block(nn.Module):

    def __init__(self, backcast_length, forecast_length, hidden_width, basis, degree, harmonics):
        super().__init__()
        self.backcast_length, self.forecast_length = (backcast_length, forecast_length)
        self.basis, self.degree, self.harmonics = (basis, degree, harmonics)
        self.fc = nn.Sequential(nn.Linear(backcast_length, hidden_width), nn.ReLU(), nn.Linear(hidden_width, hidden_width), nn.ReLU())
        count = backcast_length + forecast_length if basis == 'generic' else degree + 1 if basis == 'trend' else 2 * harmonics
        self.theta = nn.Linear(hidden_width, count)
        if basis == 'generic':
            self.register_buffer('back_basis', torch.eye(backcast_length), persistent=False)
            self.register_buffer('forecast_basis', torch.eye(forecast_length), persistent=False)
        elif basis == 'trend':
            self.register_buffer('back_basis', self._poly(backcast_length, degree), persistent=False)
            self.register_buffer('forecast_basis', self._poly(forecast_length, degree), persistent=False)
        else:
            self.register_buffer('back_basis', self._fourier(backcast_length, harmonics), persistent=False)
            self.register_buffer('forecast_basis', self._fourier(forecast_length, harmonics), persistent=False)

    @staticmethod
    def _poly(n, d):
        t = torch.linspace(-1, 0, n)
        return torch.stack([t ** i for i in range(d + 1)])

    @staticmethod
    def _fourier(n, h):
        t = torch.arange(n, dtype=torch.float32) / n
        return torch.stack([f(t) for k in range(1, h + 1) for f in (lambda x, k=k: torch.cos(2 * math.pi * k * x), lambda x, k=k: torch.sin(2 * math.pi * k * x))])

    def forward(self, x):
        theta = self.theta(self.fc(x))
        if self.basis == 'generic':
            back, forecast = (theta[:, :self.backcast_length], theta[:, self.backcast_length:])
        else:
            back, forecast = (theta @ self.back_basis, theta @ self.forecast_basis)
        return (back, forecast)

class NBEATS(nn.Module):
    architecture_id = 'nbeats'

    def __init__(self, backcast_length=24, forecast_length=6, hidden_width=128, stacks=2, blocks_per_stack=2, basis='generic', degree=2, harmonics=3, dropout=0.0):
        super().__init__()
        if min(backcast_length, forecast_length, hidden_width, stacks, blocks_per_stack, degree + 1, harmonics) < 1 or basis not in ('generic', 'trend', 'seasonality') or (not 0 <= dropout < 1):
            raise ValueError('invalid NBEATS dimensions, basis, or dropout')
        self.backcast_length, self.forecast_length = (backcast_length, forecast_length)
        self.blocks = nn.ModuleList([_Block(backcast_length, forecast_length, hidden_width, basis, degree, harmonics) for _ in range(stacks * blocks_per_stack)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_components=False):
        if x.ndim == 3 and x.shape[-1] == 1:
            x = x[..., 0]
        if x.ndim != 2 or x.shape[1] != self.backcast_length or (not x.is_floating_point()):
            raise ValueError('x must be floating [B,backcast_length]')
        residual = x
        forecast = x.new_zeros((x.shape[0], self.forecast_length))
        backs = []
        for block in self.blocks:
            back, part = block(self.dropout(residual))
            residual = residual - back
            forecast = forecast + part
            backs.append(back)
        return (forecast, residual, tuple(backs)) if return_components else forecast
