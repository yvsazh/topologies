import torch
from torch import nn


def moving_average(series, window):
    if series.ndim < 2:
        raise ValueError("series must have time and feature dimensions")
    if window < 1:
        raise ValueError("window must be positive")
    left = (window - 1) // 2
    right = window - 1 - left
    padded = torch.nn.functional.pad(series.transpose(1, 2), (left, right), mode="replicate")
    return torch.nn.functional.avg_pool1d(padded, kernel_size=window, stride=1).transpose(1, 2)


class DLinear(nn.Module):
    architecture_id = "dlinear"
    def __init__(self, input_length, prediction_length, channels=1, moving_avg=25, individual=False):
        super().__init__()
        if min(input_length, prediction_length, channels, moving_avg) < 1:
            raise ValueError("lengths, channels, and moving_avg must be positive")
        self.input_length = input_length
        self.prediction_length = prediction_length
        self.channels = channels
        self.moving_avg = moving_avg
        self.individual = individual
        if individual:
            self.trend = nn.ModuleList(nn.Linear(input_length, prediction_length) for _ in range(channels))
            self.seasonal = nn.ModuleList(nn.Linear(input_length, prediction_length) for _ in range(channels))
        else:
            self.trend = nn.Linear(input_length, prediction_length)
            self.seasonal = nn.Linear(input_length, prediction_length)

    def forward(self, series):
        if series.ndim != 3 or series.shape[1:] != (self.input_length, self.channels):
            raise ValueError("series must have shape (batch, input_length, channels)")
        trend = moving_average(series, self.moving_avg)
        remainder = series - trend
        trend = trend.transpose(1, 2)
        remainder = remainder.transpose(1, 2)
        if self.individual:
            trend_out = torch.stack([self.trend[i](trend[:, i]) for i in range(self.channels)], dim=1)
            remainder_out = torch.stack([self.seasonal[i](remainder[:, i]) for i in range(self.channels)], dim=1)
        else:
            trend_out = self.trend(trend)
            remainder_out = self.seasonal(remainder)
        return (trend_out + remainder_out).transpose(1, 2)
