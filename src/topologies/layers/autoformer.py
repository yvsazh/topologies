import torch
from torch import nn


class AutoformerSeriesDecomposition(nn.Module):
    component_id = "autoformer"

    def __init__(self, window_size=25):
        super().__init__()
        if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size <= 0:
            raise ValueError("window_size must be a positive odd integer")
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd to preserve sequence length")
        self.window_size = window_size
        self.average = nn.AvgPool1d(window_size, stride=1)

    def forward(self, values):
        if values.ndim != 3 or values.shape[1] == 0:
            raise ValueError("values must have shape (batch, nonempty time, features)")
        padding = (self.window_size - 1) // 2
        left = values[:, :1].expand(-1, padding, -1)
        right = values[:, -1:].expand(-1, padding, -1)
        padded = torch.cat((left, values, right), dim=1)
        trend = self.average(padded.transpose(1, 2)).transpose(1, 2)
        seasonal = values - trend
        return seasonal, trend
