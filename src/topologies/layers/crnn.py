import torch
from torch import nn


class CRNN(nn.Module):
    component_id = "crnn"

    def __init__(self, feature_height, in_channels=512, hidden_size=256, num_layers=2):
        super().__init__()
        for name, value in (("in_channels", in_channels), ("feature_height", feature_height),
                            ("hidden_size", hidden_size), ("num_layers", num_layers)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.in_channels = in_channels
        self.feature_height = feature_height
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.recurrent = nn.LSTM(in_channels * feature_height, hidden_size, num_layers=num_layers,
                                 batch_first=True, bidirectional=True)

    def forward(self, x):
        if not torch.is_tensor(x) or x.ndim != 4:
            raise ValueError("x must be a 4D NCHW tensor")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"x must have {self.in_channels} channels")
        if x.shape[2] != self.feature_height:
            raise ValueError(f"x must have feature height {self.feature_height}")
        if x.shape[0] <= 0 or x.shape[3] <= 0:
            raise ValueError("x must have positive batch size and width")
        if not x.is_floating_point():
            raise ValueError("x must have a floating-point dtype")
        sequence = x.permute(0, 3, 1, 2).contiguous().reshape(x.shape[0], x.shape[3], -1)
        return self.recurrent(sequence)[0]
