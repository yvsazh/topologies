import math

import torch
from torch import nn
from torch.nn import functional as F


class GASSL(nn.Module):
    component_id = "gassl"

    def __init__(self, embedding_dim, num_clusters=100, temperature=0.2, alpha=1.0, beta=1.0):
        super().__init__()
        if (
            not isinstance(embedding_dim, int)
            or isinstance(embedding_dim, bool)
            or embedding_dim <= 0
        ):
            raise ValueError("embedding_dim must be a positive integer")
        if not isinstance(num_clusters, int) or isinstance(num_clusters, bool) or num_clusters <= 0:
            raise ValueError("num_clusters must be a positive integer")
        try:
            valid_temperature = (not isinstance(temperature, bool) and math.isfinite(temperature)
                                 and temperature > 0)
        except (TypeError, ValueError, OverflowError):
            valid_temperature = False
        if not valid_temperature:
            raise ValueError("temperature must be a finite positive number")
        for name, value in (("alpha", alpha), ("beta", beta)):
            try:
                valid_weight = (isinstance(value, (int, float)) and not isinstance(value, bool)
                                and math.isfinite(value) and value >= 0)
            except (TypeError, ValueError, OverflowError):
                valid_weight = False
            if not valid_weight:
                raise ValueError(f"{name} must be a finite nonnegative number")
        if alpha == 0 and beta == 0:
            raise ValueError("alpha and beta cannot both be zero")
        self.embedding_dim = embedding_dim
        self.num_clusters = num_clusters
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.geo_head = nn.Linear(embedding_dim, num_clusters)

    def forward(self, query, positive_key, negative_keys, cluster_labels):
        if not torch.is_tensor(query) or query.ndim != 2:
            raise ValueError("query must be a rank-2 [batch, embedding_dim] tensor")
        if query.shape[0] < 1 or query.shape[1] != self.embedding_dim:
            raise ValueError("query must have a nonempty batch and the configured embedding_dim")
        if not torch.is_tensor(positive_key) or positive_key.shape != query.shape:
            raise ValueError("positive_key must match the query shape")
        if not torch.is_tensor(negative_keys) or negative_keys.ndim != 2:
            raise ValueError("negative_keys must be a rank-2 [queue_size, embedding_dim] tensor")
        if negative_keys.shape[0] < 1 or negative_keys.shape[1] != self.embedding_dim:
            raise ValueError(
                "negative_keys must have a nonempty queue and the configured embedding_dim"
            )
        if (
            not query.is_floating_point()
            or not positive_key.is_floating_point()
            or not negative_keys.is_floating_point()
        ):
            raise ValueError(
                "query, positive_key, and negative_keys must be floating-point tensors"
            )
        if positive_key.dtype != query.dtype or negative_keys.dtype != query.dtype:
            raise ValueError("query, positive_key, and negative_keys must have the same dtype")
        if positive_key.device != query.device or negative_keys.device != query.device:
            raise ValueError("query, positive_key, and negative_keys must be on the same device")
        if not torch.is_tensor(cluster_labels):
            raise ValueError("cluster_labels must have shape [batch]")
        if cluster_labels.shape == (query.shape[0], 1):
            cluster_labels = cluster_labels.squeeze(1)
        elif cluster_labels.shape != (query.shape[0],):
            raise ValueError("cluster_labels must have shape [batch]")
        if cluster_labels.dtype != torch.long:
            raise ValueError("cluster_labels must use torch.long dtype")
        if cluster_labels.device != query.device:
            raise ValueError("cluster_labels must be on the same device as query")
        if torch.any(cluster_labels < 0) or torch.any(cluster_labels >= self.num_clusters):
            raise ValueError("cluster_labels must be in [0, num_clusters)")
        positive_logits = (query * positive_key).sum(dim=-1, keepdim=True)
        negative_logits = query @ negative_keys.transpose(0, 1)
        temporal_logits = torch.cat((positive_logits, negative_logits), dim=1) / self.temperature
        positive_targets = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
        temporal_loss = F.cross_entropy(temporal_logits, positive_targets, reduction="none")
        geo_logits = self.geo_head(query)
        geo_loss = F.cross_entropy(geo_logits, cluster_labels, reduction="none")
        return self.alpha * temporal_loss + self.beta * geo_loss
