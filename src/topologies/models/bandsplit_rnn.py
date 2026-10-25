import torch
from torch import nn


class BandSplitRNN(nn.Module):
    architecture_id = "bandsplit-rnn"
    PRESETS = {
        "tiny": {"channels": 2, "bins_per_band": 1, "hidden_size": 1},
        "small": {"channels": 32, "bins_per_band": 2, "hidden_size": 32},
    }

    def __init__(self, channels, bins_per_band, hidden_size=None):
        super().__init__()
        if not isinstance(channels, int) or isinstance(channels, bool) or channels <= 0:
            raise ValueError("channels must be a positive integer")
        if (
            not isinstance(bins_per_band, int)
            or isinstance(bins_per_band, bool)
            or bins_per_band <= 0
        ):
            raise ValueError("bins_per_band must be a positive integer")
        if channels % bins_per_band:
            raise ValueError("channels must be divisible by bins_per_band")
        if hidden_size is None:
            hidden_size = channels
        if (
            not isinstance(hidden_size, int)
            or isinstance(hidden_size, bool)
            or hidden_size <= 0
        ):
            raise ValueError("hidden_size must be a positive integer")

        self.channels = channels
        self.bins_per_band = bins_per_band
        self.num_bands = channels // bins_per_band
        self.temporal_rnn = nn.GRU(
            input_size=bins_per_band,
            hidden_size=hidden_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.temporal_projection = nn.Linear(hidden_size, bins_per_band)
        self.band_rnn = nn.GRU(
            input_size=bins_per_band,
            hidden_size=hidden_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.band_projection = nn.Linear(hidden_size, bins_per_band)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("input must be a tensor with shape (batch, time, channels)")
        if x.ndim != 3:
            raise ValueError("input must have shape (batch, time, channels)")
        if x.shape[-1] != self.channels:
            raise ValueError("input final dimension must match channels")
        if x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError("input batch and time dimensions must be nonempty")
        if not x.is_floating_point():
            raise ValueError("input must contain real floating-point values")

        batch, time, _ = x.shape
        bands = x.reshape(batch, time, self.num_bands, self.bins_per_band)
        temporal_input = bands.permute(0, 2, 1, 3).reshape(
            batch * self.num_bands, time, self.bins_per_band
        )
        temporal_output, _ = self.temporal_rnn(temporal_input)
        temporal_output = self.temporal_projection(temporal_output)
        band_sequences = temporal_output.reshape(
            batch, self.num_bands, time, self.bins_per_band
        ).permute(0, 2, 1, 3)
        band_input = band_sequences.reshape(
            batch * time, self.num_bands, self.bins_per_band
        )
        band_output, _ = self.band_rnn(band_input)
        band_output = self.band_projection(band_output)
        return band_output.reshape(batch, time, self.channels)
