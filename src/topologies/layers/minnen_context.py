from math import sqrt

import torch
from torch import nn
from torch.nn import functional as F


class MinnenContext(nn.Module):
    component_id = "minnen-context"

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int = 2,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(in_channels, int) or isinstance(in_channels, bool) or in_channels <= 0:
            raise ValueError("in_channels must be a positive integer")
        if out_channels is None:
            out_channels = 2 * in_channels
        if not isinstance(out_channels, int) or isinstance(out_channels, bool) or out_channels <= 0:
            raise ValueError("out_channels must be a positive integer")
        if type(kernel_size) is not int or kernel_size != 5:
            raise ValueError("kernel_size must be 5")
        if type(stride) is not int or stride != 1:
            raise ValueError("stride must be 1")
        if type(padding) is not int or padding != 2:
            raise ValueError("padding must be 2")
        if type(bias) is not bool or bias:
            raise ValueError("bias must be False")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 5, 5))
        nn.init.kaiming_uniform_(self.weight, a=sqrt(5))
        mask = torch.zeros(5, 5)
        mask[:2, :] = 1
        mask[2, :2] = 1
        self.register_buffer("mask", mask.reshape(1, 1, 5, 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("input must be a tensor")
        if x.ndim != 4:
            raise ValueError("input must have rank 4 with shape [batch, channels, height, width]")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"input channel count must match in_channels={self.in_channels}"
            )
        if any(size <= 0 for size in x.shape):
            raise ValueError("input dimensions must be positive")
        if not x.is_floating_point():
            raise ValueError("input must use a floating-point dtype")
        return F.conv2d(x, self.weight * self.mask, stride=1, padding=2)
