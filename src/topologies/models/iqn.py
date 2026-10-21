import math

import torch
from torch import nn
from torch.nn import functional as F


class IQN(nn.Module):
    architecture_id = "iqn"
    def __init__(self, features, actions, embedding_dim=64):
        super().__init__()
        if features <= 0 or actions <= 0 or embedding_dim <= 0:
            raise ValueError("features, actions, and embedding_dim must be positive")
        self.features = features
        self.actions = actions
        self.embedding_dim = embedding_dim
        self.feature_layer = nn.Linear(features, features)
        self.quantile_layer = nn.Linear(embedding_dim, features)
        self.value_layer = nn.Linear(features, actions)

    def forward(self, observation, tau=None):
        if observation.ndim < 1 or observation.shape[-1] != self.features:
            raise ValueError(f"Expected {self.features} observation features")
        batch_shape = observation.shape[:-1]
        if tau is None:
            tau = torch.rand(batch_shape, device=observation.device, dtype=observation.dtype)
        tau = torch.as_tensor(tau, device=observation.device, dtype=observation.dtype)
        if tau.ndim == observation.ndim and tau.shape[-1] == 1:
            tau = tau.squeeze(-1)

        if tau.ndim <= len(batch_shape):
            try:
                tau = torch.broadcast_to(tau, batch_shape)
            except RuntimeError as exc:
                raise ValueError("tau must broadcast to the observation batch shape") from exc
            sample_shape = ()
            features = F.relu(self.feature_layer(observation))
        else:
            sample_shape = tau.shape[len(batch_shape):]
            quantile_shape = (*batch_shape, *sample_shape)
            try:
                tau = torch.broadcast_to(tau, quantile_shape)
            except RuntimeError as exc:
                raise ValueError("leading tau dimensions must broadcast to the observation batch") from exc
            features = F.relu(self.feature_layer(observation))
            features = features.reshape(*batch_shape, *([1] * len(sample_shape)), self.features)
        tau = tau.clamp(0.0, 1.0)
        frequencies = torch.arange(0, self.embedding_dim, device=observation.device,
                                   dtype=observation.dtype)
        cosine = torch.cos(math.pi * tau.unsqueeze(-1) * frequencies)
        quantile_embedding = F.relu(self.quantile_layer(cosine))
        return self.value_layer(features * quantile_embedding)
