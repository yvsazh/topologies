import torch
from torch import nn


class CapsuleRouting(nn.Module):
    component_id = "capsnet"
    PRESETS = {
        "tiny": {"routing_iters": 1},
        "small": {"routing_iters": 3},
        "base": {"routing_iters": 3},
    }

    def __init__(self, routing_iters: int = 3) -> None:
        super().__init__()
        if not isinstance(routing_iters, int) or isinstance(routing_iters, bool) or routing_iters <= 0:
            raise ValueError("routing_iters must be a positive integer")
        self.routing_iters = routing_iters

    def forward(self, votes: torch.Tensor) -> torch.Tensor:
        if not isinstance(votes, torch.Tensor) or votes.ndim != 4:
            raise ValueError("votes must be a rank-4 tensor shaped [batch, sources, destinations, features]")
        if any(size <= 0 for size in votes.shape[1:]):
            raise ValueError("votes must have nonempty source, destination, and feature dimensions")
        if not votes.is_floating_point():
            raise ValueError("votes must have a floating-point dtype")
        logits = votes.new_zeros(votes.shape[:3])
        for _ in range(self.routing_iters):
            couplings = torch.softmax(logits, dim=2)
            weighted_sum = (couplings.unsqueeze(-1) * votes).sum(dim=1)
            norm = torch.linalg.vector_norm(weighted_sum, dim=-1, keepdim=True)
            output = weighted_sum * norm / (1 + norm.square())
            logits = logits + (votes * output.unsqueeze(1)).sum(dim=-1)
        return output
