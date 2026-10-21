import math
import torch
from torch import nn
import torch.nn.functional as F

class _ImageTower(nn.Module):

    def __init__(self, in_channels, width, embedding_dim):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(in_channels, width, 3, stride=2, padding=1), nn.GELU(), nn.Conv2d(width, width * 2, 3, stride=2, padding=1), nn.GELU(), nn.Conv2d(width * 2, width * 2, 3, stride=2, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width * 2, embedding_dim))

    def forward(self, images):
        return F.normalize(self.features(images), dim=-1)

class _TextTower(nn.Module):

    def __init__(self, vocab_size, context_length, width, embedding_dim, heads, layers, dropout):
        super().__init__()
        self.context_length = context_length
        self.token = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(context_length, width)
        layer = nn.TransformerEncoderLayer(width, heads, width * 4, dropout, batch_first=True, norm_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.projection = nn.Linear(width, embedding_dim)

    def forward(self, tokens, mask=None):
        b, t = tokens.shape
        if not 1 <= t <= self.context_length:
            raise ValueError('token length outside context_length')
        if mask is None:
            mask = tokens.ne(0)
        if mask.shape != tokens.shape or mask.dtype not in (torch.bool, torch.uint8):
            raise ValueError('mask must be boolean [B,T], True means valid')
        mask = mask.bool()
        if (~mask.any(dim=1)).any():
            raise ValueError('each text requires at least one valid token')
        x = self.token(tokens) + self.position(torch.arange(t, device=tokens.device))[None]
        x = self.encoder(x, src_key_padding_mask=~mask)
        pooled = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        return F.normalize(self.projection(pooled), dim=-1)

class ImageTextDualEncoder(nn.Module):
    architecture_id = 'image-text-dual-encoder'
    PRESETS = {'tiny': (16, 32, 2, 1), 'small': (32, 64, 4, 2), 'base': (64, 128, 4, 4)}

    def __init__(self, vocab_size=256, context_length=32, embedding_dim=64, in_channels=3, width=32, text_width=64, text_heads=4, text_layers=2, dropout=0.0, initial_temperature=0.07, preset=None):
        super().__init__()
        if preset is not None:
            if preset not in self.PRESETS:
                raise ValueError('preset must be tiny, small, or base')
            width, text_width, text_heads, text_layers = self.PRESETS[preset]
        if min(vocab_size, context_length, embedding_dim, in_channels, width, text_width, text_heads, text_layers) < 1 or text_width % text_heads:
            raise ValueError('dimensions must be positive and text_width divisible by text_heads')
        if not 0 <= dropout < 1 or not math.isfinite(initial_temperature) or initial_temperature <= 0:
            raise ValueError('invalid dropout or initial_temperature')
        self.image_tower = _ImageTower(in_channels, width, embedding_dim)
        self.text_tower = _TextTower(vocab_size, context_length, text_width, embedding_dim, text_heads, text_layers, dropout)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / initial_temperature)))

    def encode_image(self, images):
        if images.ndim != 4 or not images.is_floating_point() or images.shape[1] != self.image_tower.features[0].in_channels:
            raise ValueError('images must be floating [B,C,H,W] with configured C')
        if min(images.shape[-2:]) < 1:
            raise ValueError('image dimensions must be nonempty')
        return self.image_tower(images)

    def encode_text(self, tokens, mask=None):
        if tokens.ndim != 2 or tokens.dtype != torch.long:
            raise ValueError('tokens must be int64 [B,T]')
        return self.text_tower(tokens, mask)

    def forward(self, images, tokens, mask=None):
        image = self.encode_image(images)
        text = self.encode_text(tokens, mask)
        if image.shape[0] != text.shape[0]:
            raise ValueError('image and text batch sizes differ')
        return (image, text, self.logit_scale.exp() * image @ text.T)
