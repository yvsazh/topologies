import random
from numbers import Integral

import torch
from torch import nn


class _ShaDowOutput(torch.Tensor):
    def __iter__(self):
        return iter((self.selected_features, self.induced_edge_index, self.local_target))


class ShaDow(nn.Module):
    component_id = "shadow"

    def __init__(self, depth=2, fanout=5, seed=0):
        super().__init__()
        if isinstance(depth, bool) or not isinstance(depth, Integral) or depth <= 0:
            raise ValueError("depth must be a positive integer")
        if isinstance(fanout, bool) or not isinstance(fanout, Integral) or fanout <= 0:
            raise ValueError("fanout must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.depth = int(depth)
        self.fanout = int(fanout)
        self.seed = int(seed)

    def forward(self, node_features, edge_index, target):
        if not isinstance(node_features, torch.Tensor) or node_features.ndim not in (2, 3):
            raise ValueError("node_features must have shape [N, F] or [B, N, F]")
        if node_features.ndim == 3:
            batch_size, num_nodes, num_features = node_features.shape
            if batch_size < 1 or num_nodes < 1 or num_features < 1:
                raise ValueError("batched node_features must have shape [B, N, F] with positive dimensions")
            if (not isinstance(edge_index, torch.Tensor) or edge_index.ndim != 3
                    or edge_index.shape[0] != batch_size or edge_index.shape[1] != 2):
                raise ValueError("batched edge_index must have shape [B, 2, E]")
            if edge_index.dtype not in (torch.int32, torch.int64):
                raise ValueError("edge_index must have an integer dtype")
            if edge_index.device != node_features.device:
                raise ValueError("node_features and edge_index must be on the same device")
            if isinstance(target, Integral) and not isinstance(target, bool):
                targets = [int(target)] * batch_size
            elif isinstance(target, torch.Tensor) and target.ndim == 0 and target.dtype in (
                    torch.int32, torch.int64):
                targets = [int(target.item())] * batch_size
            elif (isinstance(target, torch.Tensor) and target.ndim == 1
                    and target.shape[0] == batch_size and target.dtype in (torch.int32, torch.int64)):
                targets = [int(value) for value in target.detach().cpu().tolist()]
            else:
                raise ValueError("batched target must be an integer scalar or shape [B]")
            outputs = [self.forward(node_features[index], edge_index[index], targets[index])
                       for index in range(batch_size)]
            padded_features = torch.stack([
                output.as_subclass(torch.Tensor) for output in outputs
            ])
            result = padded_features.as_subclass(_ShaDowOutput)
            result.selected_features = [output.selected_features for output in outputs]
            result.induced_edge_index = [output.induced_edge_index for output in outputs]
            result.local_target = [output.local_target for output in outputs]
            return result
        num_nodes, num_features = node_features.shape
        if num_nodes < 1 or num_features < 1:
            raise ValueError("node_features must contain at least one node and feature")
        if not isinstance(edge_index, torch.Tensor) or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if edge_index.dtype not in (torch.int32, torch.int64):
            raise ValueError("edge_index must have an integer dtype")
        if edge_index.device != node_features.device:
            raise ValueError("node_features and edge_index must be on the same device")
        if isinstance(target, Integral) and not isinstance(target, bool):
            target_index = int(target)
        elif isinstance(target, torch.Tensor) and target.ndim == 0 and target.dtype in (
                torch.int32, torch.int64):
            target_index = int(target.item())
        else:
            raise ValueError("target must be an integer scalar")
        if not 0 <= target_index < num_nodes:
            raise ValueError("target must lie in [0, N)")

        edges = edge_index.detach().to(dtype=torch.long)
        if edges.numel() and bool(((edges < 0) | (edges >= num_nodes)).any()):
            raise ValueError("edge endpoints must lie in [0, N)")
        edge_rows = edges.detach().cpu().tolist()
        edge_pairs = list(zip(edge_rows[0], edge_rows[1]))
        adjacency = [set() for _ in range(num_nodes)]
        for source, destination in edge_pairs:
            adjacency[source].add(destination)

        selected = {target_index}
        frontier = {target_index}
        rng = random.Random(self.seed)
        for _ in range(self.depth):
            next_frontier = set()
            for node in sorted(frontier):
                candidates = sorted(adjacency[node] - selected)
                if len(candidates) > self.fanout:
                    candidates = rng.sample(candidates, self.fanout)
                next_frontier.update(candidates)
            selected.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break

        selected_nodes = sorted(selected)
        local_indices = {node: local for local, node in enumerate(selected_nodes)}
        node_indices = torch.tensor(selected_nodes, dtype=torch.long, device=node_features.device)
        selected_features = node_features.index_select(0, node_indices)
        induced_edges = [(local_indices[source], local_indices[destination])
                         for source, destination in edge_pairs
                         if source in local_indices and destination in local_indices]
        if induced_edges:
            sub_edge_index = torch.tensor(
                induced_edges, dtype=torch.long, device=edge_index.device
            ).t().contiguous()
        else:
            sub_edge_index = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
        padded_features = torch.zeros_like(node_features).index_copy(
            0, node_indices, selected_features
        )
        output = padded_features.as_subclass(_ShaDowOutput)
        output.selected_features = selected_features
        output.induced_edge_index = sub_edge_index
        output.local_target = local_indices[target_index]
        return output
