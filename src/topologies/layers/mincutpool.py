import torch
from torch import nn


class MinCutPool(nn.Module):
    component_id = "mincutpool"
    PRESETS = {"tiny": {"num_clusters": 2}, "small": {"num_clusters": 8}}

    def __init__(self, num_clusters, return_auxiliary=True):
        super().__init__()
        if isinstance(num_clusters, bool) or not isinstance(num_clusters, int) or num_clusters <= 0:
            raise ValueError("num_clusters must be a positive integer")
        if not isinstance(return_auxiliary, bool):
            value_error = ValueError
            raise value_error("return_auxiliary must be a bool")
        self.num_clusters = num_clusters
        self.return_auxiliary = return_auxiliary

    def _normalize_adjacency(self, adjacency):
        degree = adjacency.sum(dim=-1)
        inverse_sqrt_degree = degree.clamp_min(torch.finfo(adjacency.dtype).tiny).rsqrt()
        inverse_sqrt_degree = inverse_sqrt_degree * (degree > 0).to(adjacency.dtype)
        return inverse_sqrt_degree.unsqueeze(-1) * adjacency * inverse_sqrt_degree.unsqueeze(-2)

    def forward(self, x, adjacency, assignment_logits):
        tensors = (x, adjacency, assignment_logits)
        if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise ValueError("x, adjacency, and assignment_logits must be torch.Tensor values")
        if any(tensor.layout != torch.strided for tensor in tensors):
            raise ValueError("x, adjacency, and assignment_logits must be dense tensors")
        if any(tensor.ndim not in (2, 3) for tensor in tensors):
            raise ValueError("x, adjacency, and assignment_logits must be rank-2 or rank-3 tensors")
        if len({tensor.ndim for tensor in tensors}) != 1:
            raise ValueError("x, adjacency, and assignment_logits must have the same rank")
        node_count = x.shape[-2]
        if node_count <= 0 or x.shape[-1] <= 0:
            raise ValueError("x must have positive node and feature dimensions")
        if adjacency.shape[-2:] != (node_count, node_count):
            raise ValueError("adjacency must have shape [N, N] matching x")
        if assignment_logits.shape[-2:] != (node_count, self.num_clusters):
            raise ValueError("assignment_logits must have shape [N, num_clusters] matching x")
        if x.ndim == 3 and not (x.shape[0] == adjacency.shape[0] == assignment_logits.shape[0]):
            raise ValueError("batched inputs must have the same graph batch size")
        if any(not tensor.is_floating_point() for tensor in tensors):
            raise ValueError("x, adjacency, and assignment_logits must use floating-point dtypes")
        if any(tensor.device != x.device or tensor.dtype != x.dtype for tensor in tensors):
            raise ValueError("x, adjacency, and assignment_logits must share dtype and device")
        if any(not torch.isfinite(tensor).all().item() for tensor in tensors):
            raise ValueError("x, adjacency, and assignment_logits must contain only finite values")
        if (adjacency < 0).any().item():
            raise ValueError("adjacency must contain non-negative edge weights")

        assignments = torch.softmax(assignment_logits, dim=-1)
        normalized_adjacency = self._normalize_adjacency(adjacency)
        normalized_degree = normalized_adjacency.sum(dim=-1)
        cut_values = assignments * (normalized_adjacency @ assignments)
        cut_volumes = assignments.square() * normalized_degree.unsqueeze(-1)
        if x.ndim == 3:
            cut_numerator = cut_values.sum(dim=(-2, -1))
            cut_denominator = cut_volumes.sum(dim=(-2, -1))
        else:
            cut_numerator = cut_values.sum()
            cut_denominator = cut_volumes.sum()
        safe_denominator = torch.where(
            cut_denominator > 0, cut_denominator, torch.ones_like(cut_denominator)
        )
        cut_loss = torch.where(
            cut_denominator > 0,
            -cut_numerator / safe_denominator,
            cut_numerator * 0,
        )

        assignment_gram = assignments.transpose(-1, -2) @ assignments
        gram_norm = torch.linalg.vector_norm(assignment_gram, dim=(-2, -1), keepdim=True)
        normalized_gram = assignment_gram / gram_norm.clamp_min(torch.finfo(x.dtype).tiny)
        identity = torch.eye(self.num_clusters, dtype=x.dtype, device=x.device)
        orthogonality_loss = torch.linalg.vector_norm(
            normalized_gram - identity / self.num_clusters**0.5,
            dim=(-2, -1),
        )

        pooled_features = assignments.transpose(-1, -2) @ x
        pooled_adjacency = assignments.transpose(-1, -2) @ normalized_adjacency @ assignments
        pooled_adjacency = pooled_adjacency - torch.diag_embed(
            torch.diagonal(pooled_adjacency, dim1=-2, dim2=-1)
        )
        pooled_normalized_adjacency = self._normalize_adjacency(pooled_adjacency)
        if not self.return_auxiliary:
            return pooled_features
        return assignments, pooled_features, pooled_normalized_adjacency, cut_loss, orthogonality_loss
