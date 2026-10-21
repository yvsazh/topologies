import torch
from torch import nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    component_id = 'patch-embedding'

    def __init__(self, in_channels, embed_dim, patch_size, allow_padding=False):
        super().__init__()
        dims = (in_channels, embed_dim, patch_size)
        if any((isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in dims)):
            raise ValueError('dimensions must be positive integers')
        self.patch_size = patch_size
        self.allow_padding = allow_padding
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)

    def forward(self, x):
        h, w = x.shape[-2:]
        p = self.patch_size
        if not self.allow_padding and (h % p or w % p):
            raise ValueError('image dimensions must divide patch_size')
        if self.allow_padding:
            x = F.pad(x, (0, -w % p, 0, -h % p))
        y = self.proj(x)
        return (y.flatten(2).transpose(1, 2), (y.shape[-2], y.shape[-1]))
