import torch
from torch import nn


class DGI(nn.Module):
    architecture_id = "dgi"
    PRESETS = {
        "tiny": {"hidden_features": 16},
        "small": {"hidden_features": 128},
        "base": {"hidden_features": 512},
    }

    def __init__(self, in_features, hidden_features=512):
        super().__init__()
        if not isinstance(in_features, int) or isinstance(in_features, bool) or in_features < 1:
            raise ValueError("in_features must be a positive integer")
        if not isinstance(hidden_features, int) or isinstance(hidden_features, bool) or hidden_features < 1:
            raise ValueError("hidden_features must be a positive integer")
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.theta = nn.Parameter(torch.empty(in_features, hidden_features))
        self.prelu = nn.PReLU(init=0.25)
        self.discriminator = nn.Parameter(torch.empty(hidden_features, hidden_features))
        nn.init.xavier_uniform_(self.theta)
        nn.init.xavier_uniform_(self.discriminator)

    def _validate_inputs(self, x, adjacency, corrupted_features=None):
        if not isinstance(x, torch.Tensor) or x.ndim not in (2, 3):
            raise ValueError("x must have shape [N,F] or [B,N,F]")
        if not x.is_floating_point():
            raise ValueError("x must have a floating point dtype")
        if x.shape[-2] < 1 or x.shape[-1] != self.in_features:
            raise ValueError("x must contain at least one node and match in_features")
        if not torch.is_tensor(adjacency):
            raise ValueError("adjacency must be a tensor")
        node_count = x.shape[-2]
        shared_adjacency = adjacency.ndim == 2 and adjacency.shape == (node_count, node_count)
        batched_adjacency = (
            x.ndim == 3
            and adjacency.ndim == 3
            and adjacency.shape == (x.shape[0], node_count, node_count)
        )
        if not shared_adjacency and not batched_adjacency:
            raise ValueError("adjacency must have shape [N,N] or [B,N,N] matching x")
        if adjacency.is_complex():
            raise ValueError("adjacency must have a real numeric dtype")
        if adjacency.device != x.device:
            raise ValueError("x and adjacency must be on the same device")
        if corrupted_features is not None:
            if not isinstance(corrupted_features, torch.Tensor) or corrupted_features.shape != x.shape:
                raise ValueError("corrupted_features must have the same shape as x")
            if not corrupted_features.is_floating_point():
                raise ValueError("corrupted_features must have a floating point dtype")
            if corrupted_features.device != x.device or corrupted_features.dtype != x.dtype:
                raise ValueError("corrupted_features must match x device and dtype")

    def _propagate(self, x, adjacency):
        node_count = x.shape[-2]
        adjacency = adjacency.to(dtype=x.dtype)
        identity = torch.eye(node_count, dtype=x.dtype, device=x.device)
        augmented = adjacency + identity
        degree = augmented.sum(dim=-1)
        inverse_sqrt_degree = degree.clamp_min(torch.finfo(x.dtype).tiny).rsqrt()
        normalized = inverse_sqrt_degree.unsqueeze(-1) * augmented * inverse_sqrt_degree.unsqueeze(-2)
        return torch.matmul(normalized, torch.matmul(x, self.theta))

    def encode(self, x, adjacency):
        self._validate_inputs(x, adjacency)
        return self.prelu(self._propagate(x, adjacency))

    def forward(self, x, adjacency, corrupted_features=None):
        self._validate_inputs(x, adjacency, corrupted_features)
        if corrupted_features is None:
            node_count = x.shape[-2]
            if x.ndim == 2:
                corrupted_features = x.index_select(0, torch.randperm(node_count, device=x.device))
            else:
                corrupted_features = torch.stack(
                    [
                        graph.index_select(0, torch.randperm(node_count, device=x.device))
                        for graph in x
                    ]
                )
        positive = self.encode(x, adjacency)
        negative = self.encode(corrupted_features, adjacency)
        summary = torch.sigmoid(positive.mean(dim=-2))
        positive_logits = torch.einsum("...nh,hk,...k->...n", positive, self.discriminator, summary)
        negative_logits = torch.einsum("...nh,hk,...k->...n", negative, self.discriminator, summary)
        return torch.cat((positive_logits, negative_logits), dim=-1)
