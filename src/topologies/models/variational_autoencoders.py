import torch
from torch import nn
from torch.nn import functional as F

class VAE(nn.Module):
    architecture_id = 'vae'
    PRESETS = {'tiny': (32,), 'small': (128, 64), 'base': (256, 128)}

    def __init__(self, in_features=784, latent_dim=16, hidden_dims=(128, 64), likelihood='bernoulli', reconstruction_activation=None, preset=None):
        super().__init__()
        if preset is not None:
            if preset not in self.PRESETS:
                raise ValueError('unknown preset')
            hidden_dims = self.PRESETS[preset]
        if in_features <= 0 or latent_dim <= 0 or (not hidden_dims) or any((int(w) <= 0 for w in hidden_dims)):
            raise ValueError('dimensions must be positive')
        if likelihood not in ('bernoulli', 'gaussian'):
            raise ValueError('likelihood must be bernoulli or gaussian')
        if likelihood == 'bernoulli' and reconstruction_activation not in (None, 'sigmoid'):
            raise ValueError('Bernoulli likelihood requires sigmoid reconstruction')
        self.in_features, self.latent_dim, self.likelihood = (int(in_features), int(latent_dim), likelihood)
        enc = []
        d = in_features
        for w in hidden_dims:
            enc.extend([nn.Linear(d, int(w)), nn.ReLU()])
            d = int(w)
        self.encoder = nn.Sequential(*enc)
        self.mu = nn.Linear(d, latent_dim)
        self.logvar = nn.Linear(d, latent_dim)
        dec = []
        d = latent_dim
        for w in reversed(hidden_dims):
            dec.extend([nn.Linear(d, int(w)), nn.ReLU()])
            d = int(w)
        dec.append(nn.Linear(d, in_features))
        if reconstruction_activation == 'sigmoid':
            dec.append(nn.Sigmoid())
        elif reconstruction_activation not in (None, 'identity'):
            raise ValueError('invalid reconstruction_activation')
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError('input shape mismatch')
        h = self.encoder(x)
        return (self.mu(h), self.logvar(h))

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z):
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError('latent shape mismatch')
        return self.decoder(z)

    def sample(self, num_samples, device=None):
        if num_samples <= 0:
            raise ValueError('num_samples must be positive')
        p = next(self.parameters())
        return self.decode(torch.randn(num_samples, self.latent_dim, device=device or p.device, dtype=p.dtype))

    def forward(self, x, sample=True):
        mu, lv = self.encode(x)
        z = self.reparameterize(mu, lv) if sample else mu
        return (self.decode(z), mu, lv)

class ConvVAE(nn.Module):
    architecture_id = 'conv-vae'
    PRESETS = {'tiny': (8, 16), 'small': (16, 32, 64), 'base': (32, 64, 128)}

    def __init__(self, in_channels=1, image_size=(28, 28), latent_channels=8, features=(16, 32), likelihood='bernoulli', reconstruction_activation=None, preset=None):
        super().__init__()
        if preset is not None:
            if preset not in self.PRESETS:
                raise ValueError('unknown preset')
            features = self.PRESETS[preset]
        self.image_size = tuple(image_size)
        self.in_channels = int(in_channels)
        self.latent_channels = int(latent_channels)
        if len(self.image_size) != 2 or min(self.image_size) <= 0 or self.in_channels <= 0 or (self.latent_channels <= 0) or (not features) or any((int(c) <= 0 for c in features)):
            raise ValueError('invalid geometry')
        if likelihood not in ('bernoulli', 'gaussian'):
            raise ValueError('invalid likelihood')
        if likelihood == 'bernoulli' and reconstruction_activation not in (None, 'sigmoid'):
            raise ValueError('Bernoulli likelihood requires sigmoid reconstruction')
        self.likelihood = likelihood
        enc = []
        c = in_channels
        for f in features:
            enc.extend([nn.Conv2d(c, int(f), 4, 2, 1), nn.ReLU()])
            c = int(f)
        enc.append(nn.Conv2d(c, latent_channels, 3, padding=1))
        self.encoder = nn.Sequential(*enc)
        with torch.device('meta'):
            s = self.encoder(torch.empty(1, in_channels, *self.image_size)).shape[1:]
        self._shape = tuple(s)
        n = latent_channels * s[1] * s[2]
        self.mu = nn.Linear(n, n)
        self.logvar = nn.Linear(n, n)
        self.decoder = nn.Sequential(nn.Conv2d(latent_channels, features[-1], 3, padding=1), nn.ReLU(), nn.Conv2d(features[-1], in_channels, 3, padding=1))
        self.reconstruction_activation = nn.Sigmoid() if reconstruction_activation == 'sigmoid' or likelihood == 'bernoulli' else nn.Identity()

    def encode(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels or tuple(x.shape[-2:]) != self.image_size:
            raise ValueError('input geometry mismatch')
        h = self.encoder(x)
        return (self.mu(h.flatten(1)), self.logvar(h.flatten(1)))

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z):
        if z.ndim != 2 or z.shape[1] != self.mu.out_features:
            raise ValueError('latent shape mismatch')
        x = z.reshape(z.shape[0], *self._shape)
        x = F.interpolate(x, size=self.image_size, mode='bilinear', align_corners=False)
        return self.reconstruction_activation(self.decoder(x))

    def sample(self, num_samples, device=None):
        if num_samples <= 0:
            raise ValueError('num_samples must be positive')
        p = next(self.parameters())
        return self.decode(torch.randn(num_samples, self.mu.out_features, device=device or p.device, dtype=p.dtype))

    def forward(self, x, sample=True):
        mu, lv = self.encode(x)
        return (self.decode(self.reparameterize(mu, lv) if sample else mu), mu, lv)
