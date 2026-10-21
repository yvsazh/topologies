

import torch
from torch import nn


def entmax15(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    dim %= logits.ndim
    scaled = logits / 2
    sorted_logits = torch.sort(scaled, dim=dim, descending=True).values
    size = logits.size(dim)
    shape = [1] * logits.ndim
    shape[dim] = size
    ranks = torch.arange(1, size + 1, device=logits.device, dtype=logits.dtype).view(shape)
    cumulative = sorted_logits.cumsum(dim)
    cumulative_square = sorted_logits.square().cumsum(dim)
    mean = cumulative / ranks
    mean_square = cumulative_square / ranks
    variance_term = ranks * (mean_square - mean.square())
    delta = (1 - variance_term) / ranks
    tau = mean - delta.clamp_min(0).sqrt()
    support = tau <= sorted_logits
    support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau_star = tau.gather(dim, support_size - 1)
    return (scaled - tau_star).clamp_min(0).square()


class ObliviousDecisionTree(nn.Module):
    def __init__(self, input_dim: int, depth: int, output_dim: int):
        super().__init__()
        if input_dim < 1 or depth < 1 or output_dim < 1:
            raise ValueError("input_dim, depth, and output_dim must be positive")
        self.input_dim = input_dim
        self.depth = depth
        self.feature_logits = nn.Parameter(torch.zeros(depth, input_dim))
        self.thresholds = nn.Parameter(torch.zeros(depth))
        self.log_temperatures = nn.Parameter(torch.zeros(depth))
        self.leaves = nn.Parameter(torch.empty(2**depth, output_dim))
        nn.init.xavier_uniform_(self.leaves)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[-1] != self.input_dim:
            raise ValueError(f"expected inputs shaped (batch, {self.input_dim})")
        feature_weights = entmax15(self.feature_logits, dim=-1)
        selected = inputs @ feature_weights.transpose(0, 1)
        scaled_difference = (selected - self.thresholds) / self.log_temperatures.exp()
        choice = entmax15(
            torch.stack((scaled_difference, torch.zeros_like(scaled_difference)), dim=-1),
            dim=-1,
        )[..., 0]
        branch_probs = torch.stack((choice, 1 - choice), dim=-1)
        path_probs = inputs.new_ones((inputs.shape[0], 1))
        for level in range(self.depth):
            probs = branch_probs[:, level, :]
            path_probs = (path_probs.unsqueeze(-1) * probs.unsqueeze(1)).flatten(1)
        return path_probs @ self.leaves


class NODEEnsemble(nn.Module):
    component_id = "node-ensemble"

    def __init__(self, input_dim: int, output_dim: int, depth: int = 3, num_trees: int = 2):
        super().__init__()
        if num_trees < 1:
            raise ValueError("num_trees must be positive")
        self.trees = nn.ModuleList(
            [ObliviousDecisionTree(input_dim, depth, output_dim) for _ in range(num_trees)]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.stack([tree(inputs) for tree in self.trees], dim=0).mean(dim=0)
