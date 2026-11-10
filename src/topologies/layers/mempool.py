import math

import torch
from torch import nn


class MemPool(nn.Module):
    component_id = "mempool"

    def __init__(
        self,
        input_dim,
        output_dim,
        num_clusters,
        num_heads=5,
        tau=1.0,
        negative_slope=0.2,
    ):
        super().__init__()
        dimensions = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "num_clusters": num_clusters,
            "num_heads": num_heads,
        }
        for name, value in dimensions.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(tau) not in (int, float):
            raise ValueError("tau must be a positive finite number")
        try:
            tau = float(tau)
        except OverflowError as exc:
            raise ValueError("tau must be a positive finite number") from exc
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("tau must be a positive finite number")
        if type(negative_slope) not in (int, float):
            raise ValueError("negative_slope must be a nonnegative finite number")
        try:
            negative_slope = float(negative_slope)
        except OverflowError as exc:
            raise ValueError("negative_slope must be a nonnegative finite number") from exc
        if not math.isfinite(negative_slope) or negative_slope < 0:
            raise ValueError("negative_slope must be a nonnegative finite number")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_clusters = num_clusters
        self.num_heads = num_heads
        self.tau = float(tau)
        self.keys = nn.Parameter(torch.empty(num_heads, num_clusters, input_dim))
        self.head_mixer = nn.Conv2d(num_heads, 1, kernel_size=1, bias=False)
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)
        nn.init.uniform_(self.keys, -1.0, 1.0)
        nn.init.xavier_uniform_(self.head_mixer.weight, gain=math.sqrt(2.0))

    def forward(self, x, valid_node_mask=None, return_assignments=False):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        unbatched = x.ndim == 2
        if unbatched:
            x = x.unsqueeze(0)
        elif x.ndim != 3:
            raise ValueError("x must have shape [nodes, input_dim] or [batch, nodes, input_dim]")
        if x.shape[0] <= 0 or x.shape[1] <= 0:
            raise ValueError("x must contain at least one graph and one node")
        if x.shape[2] != self.input_dim:
            raise ValueError("x feature width must equal input_dim")
        if not x.is_floating_point():
            raise ValueError("x must use a floating-point dtype")
        if x.device != self.keys.device or x.dtype != self.keys.dtype:
            raise ValueError("x and layer parameters must have the same device and dtype")

        mask = None
        if valid_node_mask is not None:
            if not isinstance(valid_node_mask, torch.Tensor):
                raise TypeError("valid_node_mask must be a torch.Tensor")
            if unbatched and valid_node_mask.ndim == 1:
                valid_node_mask = valid_node_mask.unsqueeze(0)
            if valid_node_mask.ndim != 2 or valid_node_mask.shape != x.shape[:2]:
                raise ValueError("valid_node_mask must have shape [nodes] or [batch, nodes]")
            if valid_node_mask.dtype != torch.bool:
                raise ValueError("valid_node_mask must use a boolean dtype")
            if valid_node_mask.device != x.device:
                raise ValueError("valid_node_mask must be on the same device as x")
            if not valid_node_mask.any(dim=1).all():
                raise ValueError("each graph must contain at least one valid node")
            mask = valid_node_mask
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)

        differences = x[:, None, :, None, :] - self.keys[None, :, None, :, :]
        squared_distances = differences.square().sum(dim=-1)
        kernels = (1.0 + squared_distances / self.tau).pow(-(self.tau + 1.0) / 2.0)
        per_head_assignments = kernels / kernels.sum(dim=-1, keepdim=True)
        head_channels = per_head_assignments.transpose(-1, -2)
        mixed_logits = self.head_mixer(head_channels).squeeze(1).transpose(-1, -2)
        assignments = torch.softmax(mixed_logits, dim=-1)
        if mask is not None:
            assignments = assignments.masked_fill(~mask.unsqueeze(-1), 0.0)

        pooled_values = assignments.transpose(1, 2) @ x
        output = self.activation(self.projection(pooled_values))
        if unbatched:
            output = output.squeeze(0)
            assignments = assignments.squeeze(0)
        if return_assignments:
            return output, assignments
        return output

    @staticmethod
    def target_distribution(assignments):
        if not isinstance(assignments, torch.Tensor):
            raise TypeError("assignments must be a torch.Tensor")
        unbatched = assignments.ndim == 2
        if unbatched:
            assignments = assignments.unsqueeze(0)
        elif assignments.ndim != 3:
            raise ValueError("assignments must have shape [nodes, clusters] or [batch, nodes, clusters]")
        if assignments.shape[0] <= 0 or assignments.shape[1] <= 0 or assignments.shape[2] <= 0:
            raise ValueError("assignments dimensions must be positive")
        if not assignments.is_floating_point():
            raise ValueError("assignments must use a floating-point dtype")
        if not torch.isfinite(assignments).all() or (assignments < 0).any():
            raise ValueError("assignments must be finite and nonnegative")

        assignments = assignments.detach()
        valid_rows = assignments.sum(dim=-1, keepdim=True) > 0
        cluster_frequency = assignments.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(assignments.dtype).tiny
        )
        scores = assignments.square() / cluster_frequency
        targets = scores / scores.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(assignments.dtype).tiny
        )
        targets = targets.masked_fill(~valid_rows, 0.0)
        if unbatched:
            targets = targets.squeeze(0)
        return targets

    @staticmethod
    def auxiliary_loss(assignments):
        if not isinstance(assignments, torch.Tensor):
            raise TypeError("assignments must be a torch.Tensor")
        unbatched = assignments.ndim == 2
        if unbatched:
            assignments = assignments.unsqueeze(0)
        elif assignments.ndim != 3:
            raise ValueError("assignments must have shape [nodes, clusters] or [batch, nodes, clusters]")
        if not assignments.is_floating_point():
            raise ValueError("assignments must use a floating-point dtype")
        targets = MemPool.target_distribution(assignments)
        if targets.ndim == 2:
            targets = targets.unsqueeze(0)
        tiny = torch.finfo(assignments.dtype).tiny
        terms = targets * (targets.clamp_min(tiny).log() - assignments.clamp_min(tiny).log())
        per_graph_loss = terms.sum(dim=(-2, -1))
        return per_graph_loss.squeeze(0) if unbatched else per_graph_loss.mean()
