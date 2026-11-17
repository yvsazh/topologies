import math

import torch
from torch import nn


class SAGPool(nn.Module):
    component_id = "sagpool"

    def __init__(self, in_channels, ratio=0.5):
        super().__init__()
        if isinstance(in_channels, bool) or not isinstance(in_channels, int) or in_channels <= 0:
            raise ValueError("in_channels must be a positive integer")
        ratio_is_numeric = type(ratio) in (int, float)
        if not ratio_is_numeric:
            raise ValueError("ratio must be a finite number in (0, 1]")
        if not math.isfinite(ratio) or ratio <= 0 or ratio > 1:
            raise ValueError("ratio must be a finite number in (0, 1]")
        self.in_channels = in_channels
        self.ratio = float(ratio)
        self.attention = nn.Parameter(torch.empty(in_channels, 1))
        nn.init.xavier_uniform_(self.attention)

    def forward(self, x, edge_index, return_details=False):
        return_details_is_boolean = type(return_details) is bool
        if not return_details_is_boolean:
            raise ValueError("return_details must be a boolean")
        if not isinstance(x, torch.Tensor) or x.ndim not in (2, 3) or x.shape[-2] < 1:
            raise ValueError("x must be a nonempty tensor with shape [N, F] or [B, N, F]")
        if not x.is_floating_point():
            raise ValueError("x must have a floating-point dtype")
        if x.shape[-1] != self.in_channels:
            raise ValueError("x feature width must match in_channels")
        batched = x.ndim == 3
        expected_edge_rank = 3 if batched else 2
        valid_edge_shape = (
            isinstance(edge_index, torch.Tensor)
            and edge_index.ndim == expected_edge_rank
            and edge_index.shape[-2] == 2
        )
        if not valid_edge_shape:
            raise ValueError("edge_index must have shape [2, E] or [B, 2, E]")
        if edge_index.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("edge_index must have an integer dtype")
        if x.device != edge_index.device:
            raise ValueError("x and edge_index must be on the same device")
        if batched:
            if x.shape[0] < 1 or edge_index.shape[0] != x.shape[0]:
                raise ValueError("batched x and edge_index must have the same nonzero batch size")
            outputs = [self.forward(x[index], edge_index[index], return_details=True)
                       for index in range(x.shape[0])]
            pooled_x = torch.stack([output[0] for output in outputs])
            if return_details:
                return pooled_x, [output[1] for output in outputs], torch.stack(
                    [output[2] for output in outputs]
                ), torch.stack([output[3] for output in outputs])
            return pooled_x

        node_count = x.shape[0]
        edges = edge_index.to(dtype=torch.long)
        if edges.numel() and bool(((edges < 0) | (edges >= node_count)).any()):
            raise ValueError("edge endpoints must lie in [0, N)")

        nodes = torch.arange(node_count, device=x.device)
        source = torch.cat((edges[0], nodes))
        target = torch.cat((edges[1], nodes))
        degree = x.new_zeros(node_count).index_add_(0, source, x.new_ones(source.shape[0]))
        normalization = degree[source].rsqrt() * degree[target].rsqrt()
        projected = x @ self.attention
        messages = projected[target] * normalization.unsqueeze(1)
        aggregated = projected.new_zeros(projected.shape).index_add_(0, source, messages)
        scores = torch.tanh(aggregated)

        retained_count = math.ceil(self.ratio * node_count)
        idx = torch.topk(scores.squeeze(1), retained_count).indices
        pooled_x = x[idx] * scores[idx]
        node_positions = torch.full((node_count,), -1, dtype=torch.long, device=x.device)
        node_positions[idx] = torch.arange(retained_count, device=x.device)
        retained_edges = (node_positions[edges[0]] >= 0) & (node_positions[edges[1]] >= 0)
        pooled_edge_index = node_positions[edges[:, retained_edges]]
        if return_details:
            return pooled_x, pooled_edge_index, scores, idx
        return pooled_x
