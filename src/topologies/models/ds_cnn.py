import torch
from torch import nn
from torch.nn import functional as F


def _batch_norm(channels):
    norm = nn.BatchNorm2d(channels, eps=0.001, momentum=0.001, affine=True)
    norm.weight.requires_grad_(False)
    return norm


def _same_conv(conv, x):
    height, width = x.shape[-2:]
    kernel_height, kernel_width = conv.kernel_size
    stride_height, stride_width = conv.stride
    output_height = (height + stride_height - 1) // stride_height
    output_width = (width + stride_width - 1) // stride_width
    padding_height = max((output_height - 1) * stride_height + kernel_height - height, 0)
    padding_width = max((output_width - 1) * stride_width + kernel_width - width, 0)
    top = padding_height // 2
    left = padding_width // 2
    x = F.pad(x, (left, padding_width - left, top, padding_height - top))
    return conv(x)


class DSCNN(nn.Module):
    architecture_id = "ds-cnn"
    PRESETS = {
        "tiny": {
            "num_classes": 12,
            "channels": (64, 64, 64, 64, 64),
            "strides": ((2, 2), (1, 1), (1, 1), (1, 1), (1, 1)),
        },
        "small": {
            "num_classes": 12,
            "channels": (172, 172, 172, 172, 172),
            "strides": ((2, 1), (2, 2), (1, 1), (1, 1), (1, 1)),
        },
        "base": {
            "num_classes": 12,
            "channels": (276, 276, 276, 276, 276, 276),
            "strides": ((2, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1)),
        },
    }

    def __init__(
        self,
        in_channels=1,
        num_classes=12,
        channels=(64, 64, 64, 64, 64),
        strides=((2, 2), (1, 1), (1, 1), (1, 1), (1, 1)),
    ):
        super().__init__()
        if type(in_channels) is not int or in_channels <= 0:
            raise ValueError("in_channels must be a positive integer")
        if type(num_classes) is not int or num_classes <= 1:
            raise ValueError("num_classes must be an integer greater than one")
        if not isinstance(channels, (tuple, list)) or len(channels) < 2:
            raise ValueError("channels must contain a stem width and at least one block width")
        if any(type(width) is not int or width <= 0 for width in channels):
            raise ValueError("channel widths must be positive integers")
        if not isinstance(strides, (tuple, list)) or len(strides) != len(channels):
            raise ValueError("strides must contain one pair for the stem and each block")
        checked_strides = []
        for stride in strides:
            if not isinstance(stride, (tuple, list)) or len(stride) != 2:
                raise ValueError("each stride must be a pair of positive integers")
            if any(type(value) is not int or value <= 0 for value in stride):
                raise ValueError("each stride must be a pair of positive integers")
            checked_strides.append(tuple(stride))

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.channels = tuple(channels)
        self.strides = tuple(checked_strides)
        self.stem = nn.Conv2d(
            in_channels,
            self.channels[0],
            kernel_size=(10, 4),
            stride=self.strides[0],
            bias=False,
        )
        self.stem_norm = _batch_norm(self.channels[0])
        self.blocks = nn.ModuleList()
        for index in range(1, len(self.channels)):
            input_width = self.channels[index - 1]
            output_width = self.channels[index]
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "depthwise": nn.Conv2d(
                            input_width,
                            input_width,
                            kernel_size=3,
                            stride=self.strides[index],
                            groups=input_width,
                            bias=False,
                        ),
                        "depthwise_norm": _batch_norm(input_width),
                        "pointwise": nn.Conv2d(input_width, output_width, kernel_size=1, bias=False),
                        "pointwise_norm": _batch_norm(output_width),
                    }
                )
            )
        self.classifier = nn.Linear(self.channels[-1], num_classes)

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError("input must have shape [batch, in_channels, time, frequency]")
        x = F.relu(self.stem_norm(_same_conv(self.stem, x)))
        for block in self.blocks:
            x = F.relu(block["depthwise_norm"](_same_conv(block["depthwise"], x)))
            x = F.relu(block["pointwise_norm"](block["pointwise"](x)))
        x = x.mean(dim=(-2, -1))
        return self.classifier(x)
