import torch


class SGCPrecompute(torch.nn.Module):
    def __init__(self, k):
        super().__init__()
        if not isinstance(k, int) or isinstance(k, bool) or k < 0:
            raise ValueError("k must be a non-negative integer")
        self.k = k

    def forward(self, x, adjacency):
        if x.ndim != 2:
            raise ValueError("x must have shape (nodes, features)")
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency must be square")
        if adjacency.shape[0] != x.shape[0]:
            raise ValueError("adjacency and x must have the same node count")
        if adjacency.device != x.device:
            raise ValueError("adjacency and x must be on the same device")
        if adjacency.dtype != x.dtype:
            adjacency = adjacency.to(dtype=x.dtype)
        identity = torch.eye(adjacency.shape[0], device=x.device, dtype=x.dtype)
        with_self_loops = adjacency + identity
        degrees = with_self_loops.sum(dim=1)
        inv_sqrt_degrees = torch.where(degrees > 0, degrees.rsqrt(), torch.zeros_like(degrees))
        normalized = inv_sqrt_degrees[:, None] * with_self_loops * inv_sqrt_degrees[None, :]
        propagated = x
        for _ in range(self.k):
            propagated = normalized @ propagated
        return propagated
