import torch
from torch import nn
import torch.nn.functional as F

class _DoubleConv(nn.Module):

    def __init__(self, cin, cout, dropout=0.0):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.Dropout2d(dropout) if dropout else nn.Identity())

    def forward(self, x):
        return self.layers(x)

class UNet(nn.Module):
    architecture_id = 'unet'

    def __init__(self, in_channels=1, out_channels=2, features=(32, 64, 128, 256), dropout=0.0):
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or (not features) or any((int(c) <= 0 for c in features)):
            raise ValueError('channels and every feature width must be positive')
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0,1)')
        self.in_channels, self.out_channels = (int(in_channels), int(out_channels))
        widths = tuple(map(int, features))
        self.encoder = nn.ModuleList()
        prev = self.in_channels
        for width in widths:
            self.encoder.append(_DoubleConv(prev, width, dropout))
            prev = width
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _DoubleConv(widths[-1], widths[-1] * 2, dropout)
        self.upconvs, self.decoder = (nn.ModuleList(), nn.ModuleList())
        prev = widths[-1] * 2
        for width in reversed(widths):
            self.upconvs.append(nn.ConvTranspose2d(prev, width, 2, stride=2))
            self.decoder.append(_DoubleConv(width * 2, width, dropout))
            prev = width
        self.head = nn.Conv2d(widths[0], self.out_channels, 1)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f'expected [B,{self.in_channels},H,W] image tensor')
        skips = []
        for enc in self.encoder:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upconvs, self.decoder, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = dec(torch.cat((skip, x), dim=1))
        return self.head(x)
