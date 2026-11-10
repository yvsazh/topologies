import math
from numbers import Real

import torch
from torch import nn


class MemAE(nn.Module):
    component_id = "memae"

    def __init__(self, memory_size, feature_dim, shrinkage_ratio=1.0, epsilon=1e-12):
        super().__init__()
        if isinstance(memory_size, bool) or not isinstance(memory_size, int) or memory_size <= 0:
            raise ValueError("memory_size must be a positive integer")
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer")
        if (
            isinstance(shrinkage_ratio, bool)
            or not isinstance(shrinkage_ratio, Real)
            or not math.isfinite(shrinkage_ratio)
            or not 1.0 <= shrinkage_ratio <= 3.0
        ):
            raise ValueError("shrinkage_ratio must be a finite real in [1, 3]")
        if (
            isinstance(epsilon, bool)
            or not isinstance(epsilon, Real)
            or not math.isfinite(epsilon)
            or epsilon <= 0
        ):
            raise ValueError("epsilon must be a finite positive real")
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.shrinkage_ratio = float(shrinkage_ratio)
        self.epsilon = float(epsilon)

    def forward(self, query, memory):
        if not isinstance(query, torch.Tensor) or not isinstance(memory, torch.Tensor):
            raise TypeError("query and memory must be tensors")
        if (
            query.ndim != 2
            or query.shape[0] == 0
            or query.shape[1] != self.feature_dim
        ):
            raise ValueError("query must have shape [B, feature_dim] with a nonempty batch")
        if memory.ndim == 2:
            if memory.shape != (self.memory_size, self.feature_dim):
                raise ValueError("memory must have shape [memory_size, feature_dim]")
            memory_batch = memory.unsqueeze(0).expand(query.shape[0], -1, -1)
        elif memory.ndim == 3:
            if memory.shape != (query.shape[0], self.memory_size, self.feature_dim):
                raise ValueError("batched memory must have shape [B, memory_size, feature_dim]")
            memory_batch = memory
        else:
            raise ValueError("memory must have shape [memory_size, feature_dim] or [B, memory_size, feature_dim]")
        if not query.is_floating_point() or not memory.is_floating_point():
            raise ValueError("query and memory must have floating-point dtypes")
        if query.dtype != memory.dtype:
            raise ValueError("query and memory must have the same dtype")
        if query.device != memory.device:
            raise ValueError("query and memory must be on the same device")
        if not torch.isfinite(query).all().item() or not torch.isfinite(memory).all().item():
            raise ValueError("query and memory must contain only finite values")

        compute_dtype = query.dtype
        if compute_dtype not in (torch.float32, torch.float64):
            compute_dtype = torch.float32
        if self.epsilon < torch.finfo(compute_dtype).tiny:
            compute_dtype = torch.float64
        epsilon = torch.as_tensor(self.epsilon, dtype=compute_dtype, device=query.device)
        if epsilon.item() <= 0:
            raise ValueError("epsilon is not representable in the computation dtype")

        query_compute = query.to(dtype=compute_dtype)
        memory_compute = memory_batch.to(dtype=compute_dtype)
        query_norm = torch.linalg.vector_norm(query_compute, dim=-1, keepdim=True)
        memory_norm = torch.linalg.vector_norm(memory_compute, dim=-1)
        if (
            (query_norm == 0).any().item()
            or (memory_norm == 0).any().item()
            or not torch.isfinite(query_norm).all().item()
            or not torch.isfinite(memory_norm).all().item()
        ):
            raise ValueError("query and memory vectors must have finite nonzero norms")

        similarities = torch.matmul(query_compute.unsqueeze(1), memory_compute.transpose(1, 2)).squeeze(1)
        similarities = similarities / (query_norm * memory_norm)
        addressing = torch.softmax(similarities, dim=-1)
        threshold = self.shrinkage_ratio / self.memory_size
        difference = addressing - threshold
        shrunk = torch.relu(difference) * addressing / (difference.abs() + epsilon)
        normalization = shrunk.sum(dim=-1, keepdim=True)
        if not torch.isfinite(normalization).all().item() or (normalization <= 0).any().item():
            raise ValueError("shrinkage removed every addressing weight for at least one query")
        weights = shrunk / normalization
        retrieved = torch.matmul(weights.unsqueeze(1), memory_compute).squeeze(1).to(dtype=query.dtype)
        retrieved.weights = weights
        return retrieved
