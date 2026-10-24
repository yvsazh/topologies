import torch
from torch import nn
from torch.nn import functional as F


class ShallowMLPAutoencoder(nn.Module):
    architecture_id = "autoencoder-mlp"
    PRESETS = {
        "tiny": {"in_features": 784, "hidden_features": 108, "code_features": 6},
        "small": {"in_features": 784, "hidden_features": 532, "code_features": 6},
    }

    def __init__(self, in_features=784, hidden_features=532, code_features=6):
        super().__init__()
        dimensions = {
            "in_features": in_features,
            "hidden_features": hidden_features,
            "code_features": code_features,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.code_features = code_features
        self.encoder_hidden = nn.Linear(in_features, hidden_features)
        self.encoder_code = nn.Linear(hidden_features, code_features)
        self.decoder_hidden = nn.Linear(code_features, hidden_features)
        self.decoder_output = nn.Linear(hidden_features, in_features)

    def encode(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 2:
            raise ValueError("input must be a 2D tensor shaped [B, in_features]")
        if x.shape[1] != self.in_features:
            raise ValueError(f"input feature dimension must be {self.in_features}")
        hidden = torch.sigmoid(self.encoder_hidden(x))
        return self.encoder_code(hidden)

    def decode(self, z):
        if not isinstance(z, torch.Tensor) or z.ndim != 2:
            raise ValueError("code must be a 2D tensor shaped [B, code_features]")
        if z.shape[1] != self.code_features:
            raise ValueError(f"code dimension must be {self.code_features}")
        hidden = torch.sigmoid(self.decoder_hidden(z))
        return torch.sigmoid(self.decoder_output(hidden))

    def reconstruction_loss(self, target, reconstruction):
        if not isinstance(target, torch.Tensor) or not isinstance(reconstruction, torch.Tensor):
            raise TypeError("target and reconstruction must be tensors")
        expected_shape = (target.shape[0], self.in_features) if target.ndim == 2 else None
        if target.ndim != 2 or reconstruction.shape != target.shape or target.shape != expected_shape:
            raise ValueError(f"target and reconstruction must share shape [B, {self.in_features}]")
        if torch.any((target < 0) | (target > 1)):
            raise ValueError("target values must be in [0, 1]")
        if torch.any((reconstruction < 0) | (reconstruction > 1)):
            raise ValueError("reconstruction values must be in [0, 1]")
        return F.binary_cross_entropy(reconstruction, target, reduction="none").sum(dim=-1)

    def forward(self, x):
        return self.decode(self.encode(x))
