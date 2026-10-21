import torch
from torch import nn
from torch.nn import functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        out_channels = channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return F.relu(x + residual)


class ResNetV15(nn.Module):
    architecture_id = "resnet-v15"
    component_id = "resnet-v15"
    def __init__(self, in_channels=3, bottleneck_channels=64, num_classes=1000, stride=2):
        super().__init__()
        self.bottleneck = Bottleneck(in_channels, bottleneck_channels, stride=stride)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(bottleneck_channels * Bottleneck.expansion, num_classes)

    def forward(self, x):
        x = self.bottleneck(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
