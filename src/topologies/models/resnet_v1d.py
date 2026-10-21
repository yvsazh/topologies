import torch
from torch import nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            projection = []
            if stride != 1:
                projection.append(nn.AvgPool2d(stride, stride=stride, ceil_mode=True))
            projection.extend((
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            ))
            self.downsample = nn.Sequential(*projection)

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNetV1d(nn.Module):
    architecture_id = "resnet-v1d"
    def __init__(self, in_channels=3, num_classes=1000, base_channels=64, layers=(2, 2, 2, 2), **kwargs):
        super().__init__()
        in_channels = kwargs.pop("input_channels", in_channels)
        num_classes = kwargs.pop("classes", num_classes)
        base_channels = kwargs.pop("hidden_channels", base_channels)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected keyword arguments: {unexpected}")
        if base_channels < 2 or in_channels < 1 or num_classes < 1:
            raise ValueError("channel and class counts must be positive")
        if len(layers) != 4 or any(count < 1 for count in layers):
            raise ValueError("layers must contain four positive block counts")
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels // 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, base_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.in_channels = base_channels
        self.layer1 = self._make_layer(base_channels, layers[0], 1)
        self.layer2 = self._make_layer(base_channels * 2, layers[1], 2)
        self.layer3 = self._make_layer(base_channels * 4, layers[2], 2)
        self.layer4 = self._make_layer(base_channels * 8, layers[3], 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_channels * 8, num_classes)

    def _make_layer(self, channels, blocks, stride):
        layers = [BasicBlock(self.in_channels, channels, stride)]
        self.in_channels = channels
        layers.extend(BasicBlock(channels, channels) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.stem(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.fc(torch.flatten(self.pool(x), 1))
