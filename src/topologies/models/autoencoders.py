import torch
from torch import nn

def _activation(name):
    if name is None or name == 'identity':
        return nn.Identity()
    if name == 'sigmoid':
        return nn.Sigmoid()
    if name == 'tanh':
        return nn.Tanh()
    raise ValueError("output_activation must be None, 'identity', 'sigmoid', or 'tanh'")

class Autoencoder(nn.Module):
    architecture_id = 'autoencoder'
    PRESETS = {'tiny': (32,), 'small': (128, 64), 'base': (256, 128)}

    def __init__(self, in_features=784, latent_dim=16, hidden_dims=(128, 64), output_activation=None, preset=None):
        super().__init__()
        if preset is not None:
            if preset not in self.PRESETS:
                raise ValueError('unknown preset')
            hidden_dims = self.PRESETS[preset]
        if in_features <= 0 or latent_dim <= 0 or (not hidden_dims) or any((int(w) <= 0 for w in hidden_dims)):
            raise ValueError('dimensions must be positive')
        self.in_features, self.latent_dim = (int(in_features), int(latent_dim))
        enc = []
        d = self.in_features
        for w in hidden_dims:
            enc += [nn.Linear(d, int(w)), nn.ReLU()]
            d = int(w)
        enc.append(nn.Linear(d, self.latent_dim))
        self.encoder = nn.Sequential(*enc)
        dec = []
        d = self.latent_dim
        for w in reversed(hidden_dims):
            dec += [nn.Linear(d, int(w)), nn.ReLU()]
            d = int(w)
        dec.append(nn.Linear(d, self.in_features))
        dec.append(_activation(output_activation))
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(f'expected [B,{self.in_features}]')
        return self.encoder(x)

    def decode(self, z):
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(f'expected [B,{self.latent_dim}]')
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))

class ConvAutoencoder(nn.Module):
    architecture_id = 'conv-autoencoder'
    PRESETS = {'tiny': (8, 16), 'small': (16, 32, 64), 'base': (32, 64, 128)}

    def __init__(self, in_channels=1, image_size=(28, 28), latent_channels=8, features=(16, 32), output_activation=None, preset=None):
        super().__init__()
        if preset is not None:
            if preset not in self.PRESETS:
                raise ValueError('unknown preset')
            features = self.PRESETS[preset]
        self.image_size = tuple(image_size)
        self.in_channels = int(in_channels)
        self.latent_channels = int(latent_channels)
        if len(self.image_size) != 2 or min(self.image_size) <= 0 or self.in_channels <= 0 or (self.latent_channels <= 0) or (not features) or any((int(c) <= 0 for c in features)):
            raise ValueError('invalid image geometry/channels/features')
        enc = []
        c = self.in_channels
        for f in features:
            enc.extend([nn.Conv2d(c, int(f), 4, 2, 1), nn.ReLU()])
            c = int(f)
        enc.append(nn.Conv2d(c, self.latent_channels, 3, padding=1))
        self.encoder = nn.Sequential(*enc)
        with torch.device('meta'):
            shape = self.encoder(torch.empty(1, self.in_channels, *self.image_size)).shape[1:]
        self._encoded_shape = tuple(shape)
        dec = []
        c = self.latent_channels
        rev = list(features)[::-1]
        for f in rev:
            dec.extend([nn.Conv2d(c, int(f), 3, padding=1), nn.ReLU()])
            c = int(f)
        self.decoder = nn.Sequential(*dec)
        self.output = nn.Sequential(nn.Conv2d(c, self.in_channels, 3, padding=1), _activation(output_activation))

    def encode(self, x):
        if x.ndim != 4 or tuple(x.shape[1:2]) != (self.in_channels,) or tuple(x.shape[-2:]) != self.image_size:
            raise ValueError('input must match declared [B,C,H,W] geometry')
        return self.encoder(x)

    def decode(self, z):
        if z.ndim != 4 or tuple(z.shape[1:]) != self._encoded_shape:
            raise ValueError(f'latent must have shape [B,{self._encoded_shape}]')
        x = z
        for layer in self.decoder:
            if isinstance(layer, nn.Conv2d) and layer.stride == (1, 1):
                x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
            x = layer(x)
        x = torch.nn.functional.interpolate(x, size=self.image_size, mode='bilinear', align_corners=False)
        return self.output(x)

    def forward(self, x):
        return self.decode(self.encode(x))
