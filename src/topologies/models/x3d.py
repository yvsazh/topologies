import torch
from torch import nn
from torch.nn import functional as F


def _normalization(norm_layer, channels):
    return nn.Identity() if norm_layer is None else norm_layer(channels)


def _rounded_se_channels(channels, ratio, divisor=8):
    scaled = channels * ratio
    reduced = max(divisor, int(scaled + divisor / 2) // divisor * divisor)
    if reduced < 0.9 * scaled:
        reduced += divisor
    return reduced


class SqueezeExcitation3d(nn.Module):
    def __init__(self, channels, reduced_channels):
        super().__init__()
        self.reduce = nn.Conv3d(channels, reduced_channels, kernel_size=1)
        self.expand = nn.Conv3d(reduced_channels, channels, kernel_size=1)

    def forward(self, value):
        scale = F.adaptive_avg_pool3d(value, output_size=1)
        scale = F.relu(self.reduce(scale))
        return value * torch.sigmoid(self.expand(scale))


class FactorizedBottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        expansion=2.25,
        kernel_size=(3, 3, 3),
        stride=(1, 1, 1),
        se_ratio=0.0625,
        norm_layer=nn.BatchNorm3d,
        inner_activation=nn.SiLU,
    ):
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or expansion <= 0:
            raise ValueError("channel counts and expansion must be positive")
        if se_ratio < 0:
            raise ValueError("se_ratio must be nonnegative")
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3
        if len(kernel_size) != 3 or any(size <= 0 for size in kernel_size):
            raise ValueError("kernel_size must contain three positive dimensions")
        if len(stride) != 3 or any(size <= 0 for size in stride):
            raise ValueError("stride must contain three positive dimensions")

        hidden = int(out_channels * expansion)
        if hidden <= 0:
            raise ValueError("expansion must produce at least one inner channel")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expand = nn.Conv3d(in_channels, hidden, kernel_size=1, bias=False)
        self.norm_a = _normalization(norm_layer, hidden)
        self.act_a = nn.ReLU()
        self.depthwise = nn.Conv3d(
            hidden,
            hidden,
            kernel_size,
            stride=stride,
            padding=tuple(size // 2 for size in kernel_size),
            groups=hidden,
            bias=False,
        )
        self.norm_b = _normalization(norm_layer, hidden)
        if se_ratio > 0:
            reduced = _rounded_se_channels(hidden, se_ratio)
            self.se = SqueezeExcitation3d(hidden, reduced)
        else:
            self.se = nn.Identity()
        self.inner_activation = inner_activation()
        self.project = nn.Conv3d(hidden, out_channels, kernel_size=1, bias=False)
        self.norm_c = _normalization(norm_layer, out_channels)

        shortcut_needed = in_channels != out_channels or stride != (1, 1, 1)
        self.residual = not shortcut_needed
        if shortcut_needed:
            self.shortcut = nn.Conv3d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )
            self.shortcut_norm = (
                _normalization(norm_layer, out_channels)
                if in_channels != out_channels
                else nn.Identity()
            )
        else:
            self.shortcut = None
            self.shortcut_norm = nn.Identity()
        self.output_activation = nn.ReLU()

    def forward(self, clip):
        value = self.act_a(self.norm_a(self.expand(clip)))
        value = self.depthwise(value)
        value = self.norm_b(value)
        value = self.inner_activation(self.se(value))
        value = self.norm_c(self.project(value))
        identity = clip if self.shortcut is None else self.shortcut_norm(self.shortcut(clip))
        return self.output_activation(value + identity)


class X3D(nn.Module):
    architecture_id = "x3d"
    PRESETS = {"tiny": {"in_channels": 3, "num_classes": 2, "width": 4, "expansion": 2}}

    def __init__(
        self,
        in_channels=3,
        num_classes=10,
        width=16,
        expansion=2.25,
        depth=2,
        kernel_size=(3, 3, 3),
        se_ratio=0.0625,
    ):
        super().__init__()
        if min(in_channels, num_classes, width, depth) <= 0 or expansion <= 0:
            raise ValueError("dimensions, expansion, and depth must be positive")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.width = width
        self.stem = nn.Sequential(
            nn.Conv3d(
                in_channels,
                width,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                bias=False,
            ),
            nn.Conv3d(
                width,
                width,
                kernel_size=(5, 1, 1),
                padding=(2, 0, 0),
                groups=width,
                bias=False,
            ),
            nn.BatchNorm3d(width),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            *(
                FactorizedBottleneck(
                    width,
                    width,
                    expansion=expansion,
                    kernel_size=kernel_size,
                    se_ratio=se_ratio,
                )
                for _ in range(depth)
            )
        )
        self.classifier = nn.Linear(width, num_classes)

    def forward(self, clip):
        clip = torch.as_tensor(clip)
        if clip.ndim != 5 or clip.shape[1] != self.in_channels:
            raise ValueError("clip must have shape (batch, channels, frames, height, width)")
        value = self.blocks(self.stem(clip))
        value = value.mean(dim=(2, 3, 4))
        return self.classifier(value)
