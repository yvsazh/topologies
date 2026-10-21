import torch
from torch import nn

class CompactConvNet(nn.Module):
    architecture_id = 'compact-convnet'
    PRESETS = {'tiny': {'widths': (16, 32)}, 'small': {'widths': (32, 64)}, 'base': {'widths': (64, 128)}, 'lenet-style': {'widths': (6, 16)}}

    def __init__(self, in_channels=1, num_classes=10, widths=(32, 64), dropout=0.0):
        super().__init__()
        if type(in_channels) is not int or type(num_classes) is not int or in_channels < 1 or (num_classes < 1):
            raise ValueError('in_channels and num_classes must be positive integers')
        if not isinstance(widths, (tuple, list)) or not widths or any((type(w) is not int or w < 1 for w in widths)):
            raise ValueError('widths must contain positive integers')
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0, 1)')
        self.in_channels, self.num_classes = (in_channels, num_classes)
        blocks = []
        c = in_channels
        for w in widths:
            blocks += [nn.Conv2d(c, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=False), nn.MaxPool2d(2, ceil_mode=True)]
            c = w
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(c, num_classes)

    def forward_features(self, x):
        self._check(x)
        return self.pool(self.features(x)).flatten(1)

    def _check(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 4 or x.shape[1] != self.in_channels or (min(x.shape[-2:]) < 1):
            raise ValueError(f'expected [B,{self.in_channels},H,W] image tensor')
        if not x.is_floating_point():
            raise ValueError('images must be floating point')

    def forward(self, x):
        return self.classifier(self.dropout(self.forward_features(x)))
