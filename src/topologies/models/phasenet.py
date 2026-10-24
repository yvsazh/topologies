import math

import torch
from torch import nn
from torch.nn import functional as F


def _value_error(message):
    return ValueError(message)


class PhaseNet(nn.Module):
    architecture_id = "phasenet"
    PRESETS = {
        "tiny": {"filters_root": 2, "dropout": 0.0},
        "small": {"filters_root": 4, "dropout": 0.0},
        "base": {"filters_root": 8, "dropout": 0.0},
    }

    def __init__(self, filters_root=8, dropout=0.0):
        super().__init__()
        if isinstance(filters_root, bool) or not isinstance(filters_root, int) or filters_root < 1:
            raise ValueError("filters_root must be a positive integer")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise _value_error("dropout must be a number in [0, 1)")
        if not math.isfinite(dropout) or dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be a number in [0, 1)")

        widths = [filters_root * (2 ** depth) for depth in range(5)]
        self.input_block = self._conv_block(3, widths[0], dropout)
        self.encoder_blocks = nn.ModuleList(
            self._conv_block(widths[depth - 1], widths[depth], dropout, stride=4)
            for depth in range(1, 5)
        )
        self.upsample_blocks = nn.ModuleList(
            nn.Sequential(
                nn.ConvTranspose1d(
                    widths[depth],
                    widths[depth - 1],
                    kernel_size=7,
                    stride=4,
                    padding=3,
                    output_padding=3,
                ),
                nn.BatchNorm1d(widths[depth - 1], eps=0.001, momentum=0.01),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for depth in range(4, 0, -1)
        )
        self.decoder_blocks = nn.ModuleList(
            self._conv_block(2 * widths[depth - 1], widths[depth - 1], dropout)
            for depth in range(4, 0, -1)
        )
        self.output_head = nn.Conv1d(widths[0], 3, kernel_size=1)
        self.apply(self._initialize_weights)

    @staticmethod
    def _conv_block(in_channels, out_channels, dropout, stride=1):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride),
            nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight)
            bound = math.sqrt(6.0 / (fan_in + fan_out))
            nn.init.uniform_(module.weight, -bound, bound)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @staticmethod
    def _same_padding(x, stride):
        length = x.shape[-1]
        output_length = (length + stride - 1) // stride
        padding = max((output_length - 1) * stride + 7 - length, 0)
        left = padding // 2
        return F.pad(x, (left, padding - left))

    @staticmethod
    def _center_crop(x, length):
        extra = x.shape[-1] - length
        if extra < 0:
            raise ValueError("upsampled sequence is shorter than its skip connection")
        left = extra // 2
        return x[..., left:left + length]

    def forward_logits(self, x):
        if not isinstance(x, torch.Tensor):
            raise _value_error("input must be a torch.Tensor with 3 dimensions")
        if x.ndim != 3:
            raise ValueError("input must have 3D shape [batch, 3 channels, time]")
        if x.shape[0] < 1:
            raise ValueError("input batch dimension must be non-empty")
        if x.shape[1] != 3:
            raise ValueError("input must have exactly 3 channels")
        if x.shape[2] < 1:
            raise ValueError("input time dimension must be non-empty")
        if not x.is_floating_point():
            raise ValueError("input must use a floating-point dtype")

        features = self.input_block(self._same_padding(x, 1))
        skips = [features]
        for block in self.encoder_blocks:
            features = block(self._same_padding(features, 4))
            skips.append(features)

        for upsample, block, skip in zip(
            self.upsample_blocks, self.decoder_blocks, reversed(skips[:-1])
        ):
            features = upsample(features)
            features = self._center_crop(features, skip.shape[-1])
            features = block(self._same_padding(torch.cat((skip, features), dim=1), 1))

        return self.output_head(features)

    def forward(self, x):
        return self.forward_logits(x).softmax(dim=1)

    @staticmethod
    def soft_target_cross_entropy(logits, targets):
        if logits.ndim != 3 or logits.shape[1] != 3 or targets.shape != logits.shape:
            raise ValueError("logits and targets must have matching [batch, 3, time] shapes")
        return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
