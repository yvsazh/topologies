import math
from numbers import Real

import torch
from torch import nn


class BarlowTwins(nn.Module):
    architecture_id = "barlow-twins"
    PRESETS = {
        "tiny": {
            "in_channels": 3,
            "encoder_width": 8,
            "projector_hidden_dim": 32,
            "embedding_dim": 16,
            "lambd": 0.0051,
            "eps": 1e-5,
        },
        "small": {
            "in_channels": 3,
            "encoder_width": 32,
            "projector_hidden_dim": 256,
            "embedding_dim": 128,
            "lambd": 0.0051,
            "eps": 1e-5,
        },
    }

    def __init__(
        self,
        in_channels=3,
        encoder_width=8,
        projector_hidden_dim=32,
        embedding_dim=16,
        lambd=0.0051,
        eps=1e-5,
    ):
        super().__init__()
        dimensions = {
            "in_channels": in_channels,
            "encoder_width": encoder_width,
            "projector_hidden_dim": projector_hidden_dim,
            "embedding_dim": embedding_dim,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(lambd, bool) or not isinstance(lambd, Real) or not math.isfinite(lambd) or lambd <= 0:
            raise ValueError("lambd must be a finite positive number")
        if isinstance(eps, bool) or not isinstance(eps, Real) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be a finite positive number")

        self.in_channels = in_channels
        self.encoder_width = encoder_width
        self.projector_hidden_dim = projector_hidden_dim
        self.embedding_dim = embedding_dim
        self.lambd = float(lambd)
        self.eps = float(eps)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, encoder_width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(encoder_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(encoder_width, 2 * encoder_width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * encoder_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * encoder_width, 4 * encoder_width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * encoder_width),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
        )
        self.projector = nn.Sequential(
            nn.Linear(4 * encoder_width, projector_hidden_dim),
            nn.BatchNorm1d(projector_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projector_hidden_dim, projector_hidden_dim),
            nn.BatchNorm1d(projector_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projector_hidden_dim, embedding_dim),
        )
        self.correlation_bn = nn.BatchNorm1d(
            embedding_dim,
            eps=self.eps,
            affine=False,
            track_running_stats=False,
        )

    def _validate_views(self, view_a, view_b):
        if not torch.is_tensor(view_a) or not torch.is_tensor(view_b):
            raise ValueError("both views must be tensors")
        if view_a.ndim != 4 or view_b.ndim != 4:
            raise ValueError("both views must have shape [batch, channels, height, width]")
        if view_a.shape != view_b.shape:
            raise ValueError("both views must have the same shape")
        if view_a.shape[0] < 2:
            raise ValueError("batch size must be at least 2")
        if view_a.shape[1] != self.in_channels:
            raise ValueError(f"views must have {self.in_channels} channels")
        if view_a.shape[2] < 1 or view_a.shape[3] < 1:
            raise ValueError("view height and width must be positive")
        if view_a.device != view_b.device or view_a.dtype != view_b.dtype:
            raise ValueError("both views must have the same device and dtype")
        if not torch.is_floating_point(view_a) or not torch.is_floating_point(view_b):
            raise ValueError("both views must have floating point dtype")

    def encode_views(self, view_a, view_b):
        self._validate_views(view_a, view_b)
        return self.projector(self.encoder(view_a)), self.projector(self.encoder(view_b))

    def cross_correlation(self, embedding_a, embedding_b):
        if not torch.is_tensor(embedding_a) or not torch.is_tensor(embedding_b):
            raise ValueError("both embeddings must be tensors")
        if embedding_a.ndim != 2 or embedding_b.ndim != 2:
            raise ValueError("both embeddings must have shape [batch, features]")
        if embedding_a.shape != embedding_b.shape:
            raise ValueError("both embeddings must have the same shape")
        if embedding_a.shape[0] < 2:
            raise ValueError("batch size must be at least 2")
        if embedding_a.shape[1] != self.embedding_dim:
            raise ValueError(f"embeddings must have {self.embedding_dim} features")
        if embedding_a.device != embedding_b.device or embedding_a.dtype != embedding_b.dtype:
            raise ValueError("both embeddings must have the same device and dtype")
        if not torch.is_floating_point(embedding_a) or not torch.is_floating_point(embedding_b):
            raise ValueError("both embeddings must have floating point dtype")
        normalized_a = self.correlation_bn(embedding_a)
        normalized_b = self.correlation_bn(embedding_b)
        return normalized_a.transpose(0, 1) @ normalized_b / embedding_a.shape[0]

    def loss_from_embeddings(self, embedding_a, embedding_b):
        correlation = self.cross_correlation(embedding_a, embedding_b)
        diagonal = correlation.diagonal()
        invariance = (1 - diagonal).square().sum()
        off_diagonal = correlation - torch.diag_embed(diagonal)
        redundancy = off_diagonal.square().sum()
        return invariance + self.lambd * redundancy

    def forward(self, view_a, view_b):
        embedding_a, embedding_b = self.encode_views(view_a, view_b)
        return self.loss_from_embeddings(embedding_a, embedding_b)
