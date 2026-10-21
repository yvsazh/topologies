import torch
from torch import nn
from torch.nn import functional as F

class VQVAE(nn.Module):
    architecture_id = 'vqvae'
    PRESETS = {'tiny': (8, 16), 'small': (16, 32), 'base': (32, 64)}

    def __init__(self, in_channels=1, image_size=(28, 28), codebook_size=32, embedding_dim=16, commitment_cost=0.25, features=(16, 32), preset=None):
        super().__init__()
        if preset is not None:
            if preset not in self.PRESETS:
                raise ValueError('unknown preset')
            features = self.PRESETS[preset]
        self.in_channels = int(in_channels)
        self.image_size = tuple(image_size)
        self.codebook_size = int(codebook_size)
        self.embedding_dim = int(embedding_dim)
        self.commitment_cost = float(commitment_cost)
        if len(self.image_size) != 2 or min(self.image_size) <= 0 or min(self.in_channels, self.codebook_size, self.embedding_dim) <= 0 or (self.commitment_cost < 0) or (not features) or any((int(c) <= 0 for c in features)):
            raise ValueError('invalid VQVAE dimensions or commitment cost')
        enc = []
        c = self.in_channels
        for f in features:
            enc.extend([nn.Conv2d(c, int(f), 4, 2, 1), nn.ReLU()])
            c = int(f)
        enc.append(nn.Conv2d(c, self.embedding_dim, 1))
        self.encoder = nn.Sequential(*enc)
        self.codebook = nn.Embedding(self.codebook_size, self.embedding_dim)
        nn.init.uniform_(self.codebook.weight, -1 / self.codebook_size, 1 / self.codebook_size)
        self.decoder = nn.Sequential(nn.Conv2d(self.embedding_dim, features[-1], 3, padding=1), nn.ReLU(), nn.Conv2d(features[-1], self.in_channels, 3, padding=1))

    def encode(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels or tuple(x.shape[-2:]) != self.image_size:
            raise ValueError('input geometry mismatch')
        return self.encoder(x)

    def quantize(self, z):
        if z.ndim != 4 or z.shape[1] != self.embedding_dim:
            raise ValueError('latent must be [B,embedding_dim,H,W]')
        flat = z.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)
        emb = self.codebook.weight
        distances = flat.square().sum(1, keepdim=True) + emb.square().sum(1) - 2 * flat @ emb.t()
        indices = distances.argmin(1).view(z.shape[0], z.shape[2], z.shape[3])
        q = F.embedding(indices, self.codebook.weight).permute(0, 3, 1, 2).contiguous()
        loss = F.mse_loss(q, z.detach()) + self.commitment_cost * F.mse_loss(z, q.detach())
        return (z + (q - z).detach(), indices, loss)

    def encode_indices(self, x):
        z = self.encode(x)
        return self.quantize(z)[1]

    def decode(self, quantized):
        if quantized.ndim != 4 or quantized.shape[1] != self.embedding_dim:
            raise ValueError('quantized input geometry mismatch')
        y = self.decoder(quantized)
        return F.interpolate(y, size=self.image_size, mode='bilinear', align_corners=False)

    def decode_indices(self, indices):
        if indices.ndim != 3 or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError('indices must be integer [B,H,W]')
        if indices.numel() and (indices.min() < 0 or indices.max() >= self.codebook_size):
            raise ValueError('code index out of range')
        q = F.embedding(indices, self.codebook.weight).permute(0, 3, 1, 2).contiguous()
        return self.decode(q)

    def forward(self, x):
        z = self.encode(x)
        q, _, loss = self.quantize(z)
        return (self.decode(q), q, loss)
