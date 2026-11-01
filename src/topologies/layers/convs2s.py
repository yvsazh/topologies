import math

import torch
from torch import nn
from torch.nn import functional as F


class GatedConvolution(nn.Module):
    component_id = "convs2s"

    def __init__(self, channels=512, kernel_size=3, causal=False, dropout=0.0):
        super().__init__()
        if isinstance(channels, bool) or not isinstance(channels, int) or channels <= 0:
            raise ValueError("channels must be a positive integer")
        if (isinstance(kernel_size, bool) or not isinstance(kernel_size, int)
                or kernel_size <= 0 or kernel_size % 2 == 0):
            raise ValueError("kernel_size must be a positive odd integer")
        if type(causal) is not bool:
            raise ValueError("causal must be a boolean")
        if type(dropout) not in (int, float):
            raise ValueError("dropout must be a finite number in [0, 1)")
        if not math.isfinite(float(dropout)) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be a finite number in [0, 1)")
        self.channels = channels
        self.kernel_size = kernel_size
        self.causal = causal
        self.dropout = float(dropout)
        self.input_dropout = nn.Dropout(self.dropout)
        convolution = nn.Conv1d(channels, 2 * channels, kernel_size, bias=True)
        variance = 4.0 * (1.0 - self.dropout) / kernel_size
        nn.init.normal_(convolution.weight, mean=0.0, std=math.sqrt(variance))
        nn.init.zeros_(convolution.bias)
        self.conv = nn.utils.parametrizations.weight_norm(convolution)

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError("input must be a floating-point tensor shaped (batch, time, channels)")
        if x.shape[-1] != self.channels:
            raise ValueError(f"input must have {self.channels} channels in its last dimension")
        if not torch.is_floating_point(x):
            raise ValueError("input must be a floating-point tensor")
        left_padding = self.kernel_size - 1 if self.causal else self.kernel_size // 2
        right_padding = 0 if self.causal else self.kernel_size // 2
        values = self.input_dropout(x).transpose(1, 2)
        values = F.pad(values, (left_padding, right_padding))
        gates = F.glu(self.conv(values), dim=1).transpose(1, 2)
        return 0.5 * (x + gates)
