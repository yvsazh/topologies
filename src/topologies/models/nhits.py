import torch
from torch import nn
from torch.nn import functional as F


class _ForecastBlock(nn.Module):
    def __init__(self, input_size, horizon, hidden_size, pool_rate):
        super().__init__()
        self.pool_rate = pool_rate
        pooled_size = max(1, (input_size + pool_rate - 1) // pool_rate)
        backcast_coefficients = max(1, (input_size + pool_rate - 1) // pool_rate)
        forecast_coefficients = max(1, (horizon + pool_rate - 1) // pool_rate)
        self.backcast = nn.Sequential(
            nn.Linear(pooled_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, backcast_coefficients),
        )
        self.forecast = nn.Sequential(
            nn.Linear(pooled_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, forecast_coefficients),
        )
        self.input_size = input_size
        self.horizon = horizon

    def forward(self, residual):
        pooled = F.max_pool1d(residual.unsqueeze(1), kernel_size=self.pool_rate, stride=self.pool_rate, ceil_mode=True).squeeze(1)
        backcast_coefficients = self.backcast(pooled)
        forecast_coefficients = self.forecast(pooled)
        backcast = F.interpolate(
            backcast_coefficients.unsqueeze(1),
            size=self.input_size,
            mode="linear",
            align_corners=False,
        ).squeeze(1)
        forecast = F.interpolate(
            forecast_coefficients.unsqueeze(1),
            size=self.horizon,
            mode="linear",
            align_corners=False,
        ).squeeze(1)
        return residual - backcast, forecast


class NHiTS(nn.Module):
    architecture_id = "nhits"
    component_id = "nhits"

    def __init__(self, input_size=24, horizon=12, hidden_size=32, pool_rates=(1, 2, 4), stacks=2):
        super().__init__()
        if min(input_size, horizon, hidden_size, stacks) < 1 or not pool_rates or any(rate < 1 for rate in pool_rates):
            raise ValueError("sizes, stack count, and pool rates must be positive")
        self.input_size = input_size
        self.horizon = horizon
        self.pool_rates = tuple(pool_rates)
        self.stacks = nn.ModuleList([nn.ModuleList([_ForecastBlock(input_size, horizon, hidden_size, rate) for rate in self.pool_rates]) for _ in range(stacks)])

    def forward(self, values):
        values = torch.as_tensor(values, dtype=torch.float32)
        if values.ndim != 2 or values.shape[-1] != self.input_size:
            raise ValueError("values must have shape (batch, input_size)")
        residual = values
        forecast = values.new_zeros((values.shape[0], self.horizon))
        for stack in self.stacks:
            for block in stack:
                residual, increment = block(residual)
                forecast = forecast + increment
        return forecast
