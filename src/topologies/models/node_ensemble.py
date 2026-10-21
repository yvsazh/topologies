"""Clean-room differentiable oblivious decision trees and a small NODE ensemble."""

import torch
from torch import nn


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Project logits onto the probability simplex using sparsemax."""
    shifted = logits - logits.max(dim=dim, keepdim=True).values
    sorted_logits = torch.sort(shifted, dim=dim, descending=True).values
    size = logits.size(dim)
    ranks = torch.arange(1, size + 1, device=logits.device, dtype=logits.dtype)
    shape = [1] * logits.ndim
    shape[dim] = size
    ranks = ranks.view(shape)
    cumulative = sorted_logits.cumsum(dim)
    support = 1 + ranks * sorted_logits > cumulative
    support_size = support.sum(dim=dim, keepdim=True).clamp(min=1)
    tau = (cumulative.gather(dim, support_size - 1) - 1) / support_size.to(logits.dtype)
    return torch.clamp(shifted - tau, min=0)


class ObliviousDecisionTree(nn.Module):
    """An oblivious tree with one learned split per level and ``2**depth`` leaves."""

    def __init__(self, input_dim: int, depth: int, output_dim: int):
        super().__init__()
        if input_dim < 1 or depth < 1 or output_dim < 1:
            raise ValueError("input_dim, depth, and output_dim must be positive")
        self.input_dim = input_dim
        self.depth = depth
        self.feature_logits = nn.Parameter(torch.zeros(depth, input_dim))
        self.thresholds = nn.Parameter(torch.zeros(depth))
        self.leaves = nn.Parameter(torch.empty(2**depth, output_dim))
        nn.init.xavier_uniform_(self.leaves)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[-1] != self.input_dim:
            raise ValueError(f"expected inputs shaped (batch, {self.input_dim})")
        feature_weights = torch.softmax(self.feature_logits, dim=-1)
        selected = inputs @ feature_weights.transpose(0, 1)
        left_right = torch.stack((self.thresholds - selected, selected - self.thresholds), dim=-1)
        branch_probs = sparsemax(left_right, dim=-1)
        path_probs = inputs.new_ones((inputs.shape[0], 1))
        for level in range(self.depth):
            probs = branch_probs[:, level, :]
            path_probs = (path_probs.unsqueeze(-1) * probs.unsqueeze(1)).flatten(1)
        return path_probs @ self.leaves


class NODEEnsemble(nn.Module):
    """Two oblivious trees followed by a linear head."""

    def __init__(self, input_dim: int, output_dim: int, tree_output_dim: int = 4, depth: int = 3):
        super().__init__()
        self.trees = nn.ModuleList(
            [ObliviousDecisionTree(input_dim, depth, tree_output_dim) for _ in range(2)]
        )
        self.head = nn.Linear(2 * tree_output_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([tree(inputs) for tree in self.trees], dim=-1))
