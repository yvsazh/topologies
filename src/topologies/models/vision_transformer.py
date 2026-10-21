import torch
from torch import nn
from torch.nn import functional as F

class _Block(nn.Module):

    def __init__(self, width, heads, mlp_ratio, dropout):
        super().__init__()
        self.n1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, int(width * mlp_ratio)), nn.GELU(), nn.Dropout(dropout), nn.Linear(int(width * mlp_ratio), width), nn.Dropout(dropout))

    def forward(self, x):
        y = self.n1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        return x + self.mlp(self.n2(x))

class VisionTransformer(nn.Module):
    architecture_id = 'vision-transformer'
    PRESETS = {'tiny': {'embed_dim': 192, 'depth': 4, 'num_heads': 3}, 'small': {'embed_dim': 384, 'depth': 6, 'num_heads': 6}, 'base': {'embed_dim': 768, 'depth': 12, 'num_heads': 12}}

    def __init__(self, in_channels=3, num_classes=1000, img_size=(224, 224), patch_size=16, embed_dim=384, depth=6, num_heads=6, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        if type(in_channels) != int or type(num_classes) != int or min(in_channels, num_classes) < 1:
            raise ValueError('in_channels and num_classes must be positive integers')

        def pair(v):
            if type(v) == int:
                return (v, v)
            if isinstance(v, (tuple, list)) and len(v) == 2 and all((type(a) == int for a in v)):
                return tuple(v)
            raise ValueError('image and patch geometry must be an integer or pair')
        self.img_size, self.patch_size = (pair(img_size), pair(patch_size))
        if min(*self.img_size, *self.patch_size) < 1 or any((a < b for a, b in zip(self.img_size, self.patch_size))):
            raise ValueError('image geometry must be at least one patch')
        if any((a % b for a, b in zip(self.img_size, self.patch_size))):
            raise ValueError('img_size must be divisible by patch_size')
        if any((type(v) != int or v < 1 for v in (embed_dim, depth, num_heads))) or embed_dim % num_heads:
            raise ValueError('positive depth/width/heads required and embed_dim divisible by num_heads')
        if mlp_ratio <= 0 or not 0 <= dropout < 1:
            raise ValueError('mlp_ratio must be positive and dropout in [0,1)')
        self.in_channels, self.num_classes, self.embed_dim = (in_channels, num_classes, embed_dim)
        gh, gw = (a // b for a, b in zip(self.img_size, self.patch_size))
        self.base_grid = (gh, gw)
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, self.patch_size, stride=self.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, gh * gw + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.blocks = nn.Sequential(*[_Block(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def _position(self, gh, gw):
        if (gh, gw) == self.base_grid:
            return self.pos_embed
        cls = self.pos_embed[:, :1]
        grid = self.pos_embed[:, 1:].reshape(1, *self.base_grid, self.embed_dim).permute(0, 3, 1, 2)
        return torch.cat((cls, F.interpolate(grid, size=(gh, gw), mode='bicubic', align_corners=False).permute(0, 2, 3, 1).reshape(1, gh * gw, self.embed_dim)), 1)

    def forward_features(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 4 or x.shape[1] != self.in_channels or (min(x.shape[-2:]) < min(self.patch_size)):
            raise ValueError(f'expected [B,{self.in_channels},H,W] image tensor')
        if not x.is_floating_point():
            raise ValueError('images must be floating point')
        y = self.patch_embed(x)
        gh, gw = y.shape[-2:]
        tokens = y.flatten(2).transpose(1, 2)
        tokens = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), tokens), 1)
        return self.norm(self.blocks(self.dropout(tokens + self._position(gh, gw))))

    def forward(self, x):
        return self.head(self.forward_features(x)[:, 0])
