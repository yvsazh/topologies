import torch
from torch import nn
from torch.nn import functional as F


class _Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class _ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class UNet3Plus(nn.Module):
    architecture_id = "unet3plus"
    component_id = "unet3plus"
    def __init__(self, in_channels=1, hidden_channels=32, num_classes=2):
        super().__init__()
        if in_channels < 1 or hidden_channels < 1 or num_classes < 1:
            raise ValueError("channel counts must be positive")
        self.encoder0 = _Block(in_channels, hidden_channels)
        self.encoder1 = _Block(hidden_channels, hidden_channels * 2)
        self.encoder2 = _Block(hidden_channels * 2, hidden_channels * 4)
        self.pool = nn.MaxPool2d(2)
        self.source_same = _ConvBNReLU(hidden_channels * 2, hidden_channels)
        self.source_shallow = _ConvBNReLU(hidden_channels, hidden_channels)
        self.source_deep = _ConvBNReLU(hidden_channels * 4, hidden_channels)
        self.fuse = _ConvBNReLU(hidden_channels * 3, hidden_channels)
        self.classifier = nn.Conv2d(hidden_channels, num_classes, 1)

    def forward(self, x):
        input_size = x.shape[-2:]
        enc0 = self.encoder0(x)
        pooled0 = self.pool(enc0)
        enc1 = self.encoder1(pooled0)
        enc2 = self.encoder2(self.pool(enc1))
        same = self.source_same(enc1)
        shallow = self.source_shallow(pooled0)
        deep = self.source_deep(enc2)
        deep = F.interpolate(deep, size=enc1.shape[-2:], mode="bilinear", align_corners=False)
        fused = torch.cat((same, shallow, deep), dim=1)
        logits = self.classifier(self.fuse(fused))
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
