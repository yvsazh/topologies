import torch
from torch import nn


class APPNP(nn.Module):
    component_id = "appnp"

    def __init__(self, alpha=0.1, steps=10):
        super().__init__()
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or not 0.0 < alpha <= 1.0
        ):
            raise ValueError("alpha must be a finite number in (0, 1]")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("steps must be a positive integer")
        self.alpha = float(alpha)
        self.steps = steps

    def forward(self, h, adjacency):
        if (
            not isinstance(h, torch.Tensor)
            or h.layout != torch.strided
            or h.ndim not in (2, 3)
            or min(h.shape) <= 0
            or not h.is_floating_point()
        ):
            raise ValueError("h must be a non-empty floating tensor with shape [N, C] or [B, N, C]")
        if not isinstance(adjacency, torch.Tensor) or adjacency.ndim not in (2, 3):
            raise ValueError("adjacency must have shape [N, N] or [B, N, N]")
        nodes = h.shape[-2]
        if h.ndim == 2:
            if adjacency.ndim != 2 or adjacency.shape != (nodes, nodes):
                raise ValueError("adjacency shape must match the node count in h")
        elif adjacency.ndim == 2:
            if adjacency.shape != (nodes, nodes):
                raise ValueError("adjacency shape must match the node count in h")
        elif adjacency.shape != (h.shape[0], nodes, nodes):
            raise ValueError("batched adjacency shape must match the batch and node counts in h")
        if adjacency.layout not in (torch.strided, torch.sparse_coo, torch.sparse_csr):
            raise ValueError("adjacency must use dense, sparse COO, or sparse CSR layout")
        if adjacency.ndim == 3 and adjacency.layout != torch.strided:
            raise ValueError("batched adjacency must use a dense layout")
        if not adjacency.is_floating_point():
            raise ValueError("adjacency must use a floating-point dtype")
        if adjacency.dtype != h.dtype:
            raise ValueError("h and adjacency must have matching dtype")
        if adjacency.device != h.device:
            raise ValueError("h and adjacency must be on the same device")
        z = h
        for _ in range(self.steps):
            if adjacency.layout == torch.strided:
                propagated = adjacency @ z
            elif z.ndim == 2:
                propagated = torch.sparse.mm(adjacency, z)
            else:
                batch_size, nodes, classes = z.shape
                flattened = z.permute(1, 0, 2).reshape(nodes, batch_size * classes)
                propagated = torch.sparse.mm(adjacency, flattened)
                propagated = propagated.reshape(nodes, batch_size, classes).permute(1, 0, 2)
            z = (1.0 - self.alpha) * propagated + self.alpha * h
        return torch.softmax(z, dim=-1)
