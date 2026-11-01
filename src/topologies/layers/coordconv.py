import torch
from torch import nn


class CoordConv(nn.Module):
    component_id = "coordconv"

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        for name, value in (("in_channels", in_channels), ("out_channels", out_channels)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(bias) is not bool:
            raise ValueError("bias must be a boolean")
        self.kernel_size = self._pair(kernel_size, "kernel_size", positive=True)
        self.stride = self._pair(stride, "stride", positive=True)
        self.padding = self._pair(padding, "padding", positive=False)
        self.dilation = self._pair(dilation, "dilation", positive=True)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels + 2,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=bias,
        )

    @staticmethod
    def _pair(value, name, positive):
        if type(value) is int:
            values = (value, value)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            values = tuple(value)
        else:
            raise ValueError(f"{name} must be an integer or a pair of integers")
        lower_bound = 1 if positive else 0
        if any(type(item) is not int or item < lower_bound for item in values):
            qualifier = "positive" if positive else "nonnegative"
            raise ValueError(f"{name} values must be {qualifier} integers")
        return values

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("input must be a torch.Tensor")
        if x.ndim != 4:
            raise ValueError("input must have shape [N, C, H, W]")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"input channel count must be {self.in_channels}")
        if any(size <= 0 for size in (x.shape[0], x.shape[2], x.shape[3])):
            raise ValueError("input batch and spatial dimensions must be positive")
        if not torch.is_floating_point(x):
            raise ValueError("input must have a floating-point dtype")
        batch, _, height, width = x.shape
        if height == 1:
            rows = x.new_zeros((1, 1, height, 1))
        else:
            rows = torch.arange(height, device=x.device, dtype=x.dtype)
            rows = (2 * rows / (height - 1) - 1).reshape(1, 1, height, 1)
        if width == 1:
            columns = x.new_zeros((1, 1, 1, width))
        else:
            columns = torch.arange(width, device=x.device, dtype=x.dtype)
            columns = (2 * columns / (width - 1) - 1).reshape(1, 1, 1, width)
        rows = rows.expand(batch, 1, height, width)
        columns = columns.expand(batch, 1, height, width)
        return self.conv(torch.cat((x, rows, columns), dim=1))
