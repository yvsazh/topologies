import torch
from torch import nn


class LSTNet(nn.Module):
    architecture_id = "lstnet"
    def __init__(self, input_dim, conv_channels=16, kernel_size=3, hidden_size=16, skip_period=2, skip_hidden_size=8, highway_window=1, output_dim=None, dropout=0.0):
        super().__init__()
        if min(input_dim, conv_channels, kernel_size, hidden_size, skip_period, skip_hidden_size, highway_window) <= 0:
            raise ValueError("dimensions and skip period must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if output_dim is not None and output_dim != input_dim:
            raise ValueError("the highway path requires output_dim to equal input_dim")
        self.input_dim = input_dim
        self.skip_period = skip_period
        self.highway_window = highway_window
        self.conv = nn.Conv1d(input_dim, conv_channels, kernel_size)
        self.dropout = nn.Dropout(dropout)
        self.recurrent = nn.GRU(conv_channels, hidden_size, batch_first=True)
        self.skip_recurrent = nn.GRU(conv_channels, skip_hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size + skip_period * skip_hidden_size, input_dim)
        self.highway = nn.Linear(highway_window, 1)

    def forward(self, values):
        values = torch.as_tensor(values, dtype=self.conv.weight.dtype, device=self.conv.weight.device)
        if values.ndim != 3 or values.shape[-1] != self.input_dim:
            raise ValueError("values must have shape (batch, time, input_dim)")
        if values.shape[1] < max(self.conv.kernel_size[0], self.highway_window):
            raise ValueError("series is shorter than the convolution kernel or highway window")
        features = torch.relu(self.conv(values.transpose(1, 2))).transpose(1, 2)
        features = self.dropout(features)
        _, recurrent_state = self.recurrent(features)
        skip_steps = features.shape[1] // self.skip_period
        if skip_steps == 0:
            raise ValueError("convolution output must contain at least one full skip period")
        seasonal = features[:, -skip_steps * self.skip_period:, :]
        seasonal = seasonal.reshape(values.shape[0], skip_steps, self.skip_period, -1)
        seasonal = seasonal.permute(0, 2, 1, 3).reshape(values.shape[0] * self.skip_period, skip_steps, -1)
        _, skip_state = self.skip_recurrent(seasonal)
        skip_features = skip_state[-1].reshape(values.shape[0], self.skip_period * skip_state.shape[-1])
        recurrent_features = torch.cat((recurrent_state[-1], skip_features), dim=-1)
        nonlinear = self.output(recurrent_features)
        highway = self.highway(values[:, -self.highway_window:, :].transpose(1, 2)).squeeze(-1)
        return nonlinear + highway
