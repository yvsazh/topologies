import torch
from torch import nn

class _IR(nn.Module):

    def __init__(self, ci, co, stride, expand):
        super().__init__()
        hidden = ci * expand
        seq = []
        if expand != 1:
            seq += [nn.Conv2d(ci, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU6(inplace=False)]
        seq += [nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False), nn.BatchNorm2d(hidden), nn.ReLU6(inplace=False), nn.Conv2d(hidden, co, 1, bias=False), nn.BatchNorm2d(co)]
        self.body = nn.Sequential(*seq)
        self.use_residual = stride == 1 and ci == co

    def forward(self, x):
        y = self.body(x)
        return x + y if self.use_residual else y

class MobileNetV2(nn.Module):
    architecture_id = 'mobilenet-v2'
    SCHEDULE = ((1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1))
    PRESETS = {'tiny': {'width_multiplier': 0.25}, 'small': {'width_multiplier': 0.5}, 'base': {'width_multiplier': 1.0}}

    def __init__(self, in_channels=3, num_classes=1000, width_multiplier=1.0, dropout=0.0, round_nearest=8):
        super().__init__()
        if type(in_channels) != int or type(num_classes) != int or min(in_channels, num_classes) < 1:
            raise ValueError('in_channels and num_classes must be positive integers')
        if width_multiplier <= 0 or round_nearest < 1 or (not 0 <= dropout < 1):
            raise ValueError('invalid width_multiplier, round_nearest, or dropout')
        self.in_channels, self.num_classes = (in_channels, num_classes)

        def scale(c):
            return max(round_nearest, int(c * width_multiplier + round_nearest / 2) // round_nearest * round_nearest)
        stem = scale(32)
        self.stem = nn.Sequential(nn.Conv2d(in_channels, stem, 3, 2, 1, bias=False), nn.BatchNorm2d(stem), nn.ReLU6(inplace=False))
        c = stem
        units = []
        for t, ch, n, s in self.SCHEDULE:
            out = scale(ch)
            for j in range(n):
                units.append(_IR(c, out, s if j == 0 else 1, t))
                c = out
        last = scale(1280 if width_multiplier <= 1 else 1280 * width_multiplier)
        self.features = nn.Sequential(*units, nn.Conv2d(c, last, 1, bias=False), nn.BatchNorm2d(last), nn.ReLU6(inplace=False))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(last, num_classes)

    def _check(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 4 or x.shape[1] != self.in_channels or (min(x.shape[-2:]) < 1):
            raise ValueError(f'expected [B,{self.in_channels},H,W]')
        if not x.is_floating_point():
            raise ValueError('images must be floating point')

    def forward_features(self, x):
        self._check(x)
        return self.features(self.stem(x))

    def forward(self, x):
        return self.classifier(self.dropout(self.forward_features(x).mean((-2, -1))))
