import torch
from torch import nn


class SimpleShot(nn.Module):
    component_id = "simpleshot"

    def __init__(self, transform="center_l2"):
        super().__init__()
        if not isinstance(transform, str) or transform not in {"unnormalized", "l2", "center_l2"}:
            raise ValueError("transform must be 'unnormalized', 'l2', or 'center_l2'")
        self.transform = transform

    def forward(self, support_features, query_features, base_feature_mean=None):
        if not isinstance(support_features, torch.Tensor) or support_features.ndim != 4:
            raise ValueError("support_features must be a tensor with shape (B, C, K, D)")
        if not isinstance(query_features, torch.Tensor) or query_features.ndim != 3:
            raise ValueError("query_features must be a tensor with shape (B, Q, D)")
        if any(size == 0 for size in support_features.shape):
            raise ValueError("support_features batch, class, shot, and feature axes must be non-empty")
        if query_features.shape[0] == 0 or query_features.shape[1] == 0:
            raise ValueError("query_features batch and query axes must be non-empty")
        if support_features.shape[0] != query_features.shape[0]:
            raise ValueError("support_features and query_features batch dimensions must match")
        if support_features.shape[-1] != query_features.shape[-1]:
            raise ValueError("support_features and query_features feature dimensions must match")
        if not support_features.is_floating_point() or not query_features.is_floating_point():
            raise ValueError("support_features and query_features must have floating point dtype")
        if support_features.device != query_features.device or support_features.dtype != query_features.dtype:
            raise ValueError("support_features and query_features must have the same device and dtype")
        if not torch.isfinite(support_features).all() or not torch.isfinite(query_features).all():
            raise ValueError("support_features and query_features must contain only finite values")

        centroids = support_features.mean(dim=2)
        if self.transform == "center_l2":
            if not isinstance(base_feature_mean, torch.Tensor):
                raise ValueError("base_feature_mean is required for the center_l2 transform")
            valid_shared_mean = base_feature_mean.ndim == 1 and base_feature_mean.shape[0] == support_features.shape[-1]
            valid_batched_mean = (
                base_feature_mean.ndim == 2
                and base_feature_mean.shape == (support_features.shape[0], support_features.shape[-1])
            )
            if not valid_shared_mean and not valid_batched_mean:
                raise ValueError("base_feature_mean must have shape (D,) or (B, D)")
            if not base_feature_mean.is_floating_point():
                raise ValueError("base_feature_mean must have floating point dtype")
            if base_feature_mean.device != support_features.device or base_feature_mean.dtype != support_features.dtype:
                raise ValueError("base_feature_mean must have the same device and dtype as the features")
            if not torch.isfinite(base_feature_mean).all():
                raise ValueError("base_feature_mean must contain only finite values")
            mean = base_feature_mean.unsqueeze(1) if base_feature_mean.ndim == 2 else base_feature_mean
            centroids = centroids - mean
            query_features = query_features - mean

        if self.transform in {"l2", "center_l2"}:
            centroid_norms = torch.linalg.vector_norm(centroids, dim=-1, keepdim=True)
            query_norms = torch.linalg.vector_norm(query_features, dim=-1, keepdim=True)
            if torch.any(centroid_norms == 0) or torch.any(query_norms == 0):
                raise ValueError("normalized features must have non-zero L2 norm")
            centroids = centroids / centroid_norms
            query_features = query_features / query_norms

        distances = torch.linalg.vector_norm(
            query_features.unsqueeze(2) - centroids.unsqueeze(1), dim=-1
        )
        return -distances
