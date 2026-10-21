import torch
from torch import nn

class _Basic(nn.Module):
    expansion = 1

    def __init__(self, ci, co, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(ci, co, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(co)
        self.conv2 = nn.Conv2d(co, co, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(co)
        self.skip = nn.Identity() if stride == 1 and ci == co else nn.Sequential(nn.Conv2d(ci, co, 1, stride, bias=False), nn.BatchNorm2d(co))

    def forward(self, x):
        return torch.relu(self.bn2(self.conv2(torch.relu(self.bn1(self.conv1(x))))) + self.skip(x))

class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, ci, co, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(ci, co, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(co)
        self.conv2 = nn.Conv2d(co, co, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(co)
        self.conv3 = nn.Conv2d(co, co * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(co * 4)
        self.skip = nn.Identity() if stride == 1 and ci == co * 4 else nn.Sequential(nn.Conv2d(ci, co * 4, 1, stride, bias=False), nn.BatchNorm2d(co * 4))

    def forward(self, x):
        y = torch.relu(self.bn1(self.conv1(x)))
        y = torch.relu(self.bn2(self.conv2(y)))
        return torch.relu(self.bn3(self.conv3(y)) + self.skip(x))

class ResNet(nn.Module):
    architecture_id = 'resnet'
    PRESETS = {'tiny': {'block': 'basic', 'layers': (1, 1, 1, 1), 'widths': (16, 32, 64, 128)}, 'small': {'block': 'basic', 'layers': (2, 2, 2, 2), 'widths': (32, 64, 128, 256)}, 'base': {'block': 'bottleneck', 'layers': (3, 4, 6, 3), 'widths': (64, 128, 256, 512)}, 'resnet18': {'block': 'basic', 'layers': (2, 2, 2, 2), 'widths': (64, 128, 256, 512)}, 'resnet34': {'block': 'basic', 'layers': (3, 4, 6, 3), 'widths': (64, 128, 256, 512)}, 'resnet50': {'block': 'bottleneck', 'layers': (3, 4, 6, 3), 'widths': (64, 128, 256, 512)}}

    def __init__(self, in_channels=3, num_classes=1000, block='basic', layers=(2, 2, 2, 2), widths=(64, 128, 256, 512), dropout=0.0, small_image_stem=True):
        super().__init__()
        if type(in_channels) != int or type(num_classes) != int or min(in_channels, num_classes) < 1:
            raise ValueError('channel and class counts must be positive integers')
        if block not in ('basic', 'bottleneck') or not layers or len(layers) != len(widths) or any((type(v) != int or v < 1 for v in (*layers, *widths))):
            raise ValueError('block and matching nonempty positive stage layers/widths required')
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0, 1)')
        self.in_channels, self.num_classes = (in_channels, num_classes)
        stem_stride = 1 if small_image_stem else 2
        self.stem = nn.Sequential(nn.Conv2d(in_channels, widths[0], 7 if not small_image_stem else 3, stem_stride, 3 if not small_image_stem else 1, bias=False), nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=False), *([] if small_image_stem else [nn.MaxPool2d(3, 2, 1)]))
        typ = _Basic if block == 'basic' else _Bottleneck
        c = widths[0]
        stages = []
        for i, (depth, w) in enumerate(zip(layers, widths)):
            units = []
            for j in range(depth):
                units.append(typ(c, w, 2 if i > 0 and j == 0 else 1))
                c = w * typ.expansion
            stages.append(nn.Sequential(*units))
        self.stages = nn.ModuleList(stages)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(c, num_classes)

    def _check(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 4 or x.shape[1] != self.in_channels or (min(x.shape[-2:]) < 1):
            raise ValueError(f'expected [B,{self.in_channels},H,W]')
        if not x.is_floating_point():
            raise ValueError('images must be floating point')

    def forward_features(self, x):
        self._check(x)
        x = self.stem(x)
        out = []
        for stage in self.stages:
            x = stage(x)
            out.append(x)
        return tuple(out)

    def forward(self, x):
        return self.classifier(self.dropout(self.forward_features(x)[-1].mean((-2, -1))))
