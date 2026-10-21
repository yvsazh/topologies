import math
import torch
from torch import nn

class _Coupling(nn.Module):

    def __init__(self, features, mask, hidden_dims, scale_bound):
        super().__init__()
        self.register_buffer('mask', mask)
        self.scale_bound = float(scale_bound)
        dims = [features, *hidden_dims, 2 * features]
        layers = []
        for a, b in zip(dims, dims[1:]):
            layers.append(nn.Linear(a, b))
        self.net = nn.Sequential(*sum(([layer, nn.Tanh()] for layer in layers[:-1]), []), layers[-1])
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)

    def params(self, x):
        a = x * self.mask
        s, t = self.net(a).chunk(2, -1)
        active = 1 - self.mask
        return (self.scale_bound * torch.tanh(s) * active, t * active)

    def forward(self, x):
        s, t = self.params(x)
        y = x * self.mask + (1 - self.mask) * (x * torch.exp(s) + t)
        return (y, s.sum(-1))

    def inverse(self, y):
        s, t = self.params(y)
        x = y * self.mask + (1 - self.mask) * (y - t) * torch.exp(-s)
        return (x, -s.sum(-1))

class RealNVP(nn.Module):
    architecture_id = 'realnvp'
    PRESETS = {'tiny': (16,), 'small': (64, 64), 'base': (128, 128)}

    def __init__(self, features=2, coupling_layers=4, hidden_dims=(64, 64), scale_bound=2.0, mask_scheme='alternating', preset=None):
        super().__init__()
        if preset is not None:
            if preset not in self.PRESETS:
                raise ValueError('unknown preset')
            hidden_dims = self.PRESETS[preset]
        if features < 2 or coupling_layers <= 0 or (not hidden_dims) or any((int(x) <= 0 for x in hidden_dims)) or (scale_bound <= 0):
            raise ValueError('invalid RealNVP dimensions')
        if mask_scheme != 'alternating':
            raise ValueError('only alternating masks are supported')
        self.features = int(features)
        self.scale_bound = float(scale_bound)
        masks = [(torch.arange(features) % 2 == i).float() for i in range(2)]
        self.couplings = nn.ModuleList([_Coupling(features, masks[i % 2], hidden_dims, scale_bound) for i in range(coupling_layers)])

    def _check(self, x):
        if x.ndim != 2 or x.shape[1] != self.features:
            raise ValueError(f'expected [B,{self.features}]')

    def forward(self, x):
        self._check(x)
        total = x.new_zeros(x.shape[0])
        for layer in self.couplings:
            x, ld = layer(x)
            total = total + ld
        return (x, total)

    def inverse(self, z):
        self._check(z)
        total = z.new_zeros(z.shape[0])
        for layer in reversed(self.couplings):
            z, ld = layer.inverse(z)
            total = total + ld
        return (z, total)

    def log_prob(self, x):
        z, logdet = self(x)
        base = -0.5 * (z.square() + math.log(2 * math.pi)).sum(-1)
        return base + logdet

    def sample(self, num_samples, device=None):
        if num_samples <= 0:
            raise ValueError('num_samples must be positive')
        p = next(self.parameters())
        z = torch.randn(num_samples, self.features, device=device or p.device, dtype=p.dtype)
        return self.inverse(z)[0]
