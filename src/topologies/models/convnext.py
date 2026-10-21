import torch
from torch import nn

class _DropPath(nn.Module):

    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x * x.new_empty(shape).bernoulli_(keep) / keep

class _Block(nn.Module):

    def __init__(self, c, drop_path, layer_scale):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 7, padding=3, groups=c)
        self.norm = nn.LayerNorm(c, eps=1e-06)
        self.pw1 = nn.Linear(c, 4 * c)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(4 * c, c)
        self.gamma = nn.Parameter(torch.full((c,), layer_scale)) if layer_scale > 0 else None
        self.drop_path = _DropPath(drop_path)

    def forward(self, x):
        y = self.dw(x).permute(0, 2, 3, 1)
        y = self.pw2(self.act(self.pw1(self.norm(y))))
        if self.gamma is not None:
            y = y * self.gamma
        return x + self.drop_path(y.permute(0, 3, 1, 2))

class ConvNeXt(nn.Module):
    architecture_id = 'convnext'
    PRESETS = {'tiny': {'depths': (1, 1, 2, 1), 'widths': (32, 64, 128, 256)}, 'small': {'depths': (2, 2, 4, 2), 'widths': (48, 96, 192, 384)}, 'base': {'depths': (3, 3, 9, 3), 'widths': (96, 192, 384, 768)}}

    def __init__(self, in_channels=3, num_classes=1000, depths=(3, 3, 9, 3), widths=(96, 192, 384, 768), drop_path_rate=0.0, layer_scale=1e-06, dropout=0.0):
        super().__init__()
        if type(in_channels) != int or type(num_classes) != int or min(in_channels, num_classes) < 1:
            raise ValueError('in_channels and num_classes must be positive integers')
        if len(depths) != 4 or len(widths) != 4 or any((type(v) != int or v < 1 for v in (*depths, *widths))):
            raise ValueError('depths and widths must have four positive integers')
        if not 0 <= drop_path_rate < 1 or not 0 <= dropout < 1 or layer_scale < 0:
            raise ValueError('invalid dropout, drop_path_rate, or layer_scale')
        self.in_channels, self.num_classes = (in_channels, num_classes)
        self.downsample_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(in_channels, widths[0], 4, 4), _ToLast(), nn.LayerNorm(widths[0]), _ToFirst())])
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(_ToLast(), nn.LayerNorm(widths[i]), _ToFirst(), nn.Conv2d(widths[i], widths[i + 1], 2, 2)))
        total = sum(depths)
        rates = [drop_path_rate * i / max(total - 1, 1) for i in range(total)]
        k = 0
        self.stages = nn.ModuleList()
        for depth, width in zip(depths, widths):
            self.stages.append(nn.Sequential(*[_Block(width, rates[k + j], layer_scale) for j in range(depth)]))
            k += depth
        self.norm = nn.LayerNorm(widths[-1])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(widths[-1], num_classes)

    def _check(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 4 or x.shape[1] != self.in_channels or (min(x.shape[-2:]) < 4):
            raise ValueError(f'expected [B,{self.in_channels},H,W] with H,W >= 4')
        if not x.is_floating_point():
            raise ValueError('images must be floating point')

    def forward_features(self, x):
        self._check(x)
        feats = []
        for down, stage in zip(self.downsample_layers, self.stages):
            x = stage(down(x))
            feats.append(x)
        return tuple(feats)

    def forward(self, x):
        return self.head(self.dropout(self.norm(self.forward_features(x)[-1].mean((2, 3)))))

class _ToLast(nn.Module):

    def forward(self, x):
        return x.permute(0, 2, 3, 1)

class _ToFirst(nn.Module):

    def forward(self, x):
        return x.permute(0, 3, 1, 2)
