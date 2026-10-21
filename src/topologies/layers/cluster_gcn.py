import torch
from torch import nn


class ClusterGCN(nn.Module):
    component_id = "cluster-gcn"

    def __init__(self, *args, **kwargs):
        super().__init__()
        if args or kwargs:
            raise ValueError("ClusterGCN does not accept constructor arguments")

    def forward(self, x, edge_index, cluster_ids, cluster_id):
        if not isinstance(x, torch.Tensor) or x.ndim != 2:
            raise ValueError("x must be a tensor with shape [N, F]")
        if not isinstance(edge_index, torch.Tensor) or edge_index.ndim != 2:
            raise ValueError("edge_index must be a tensor with shape [2, E]")
        if edge_index.dtype not in (torch.int32, torch.int64):
            raise ValueError("edge_index must have an integer dtype")
        if not isinstance(cluster_ids, torch.Tensor) or cluster_ids.ndim not in (1, 2):
            raise ValueError("cluster_ids must be a tensor with shape [N] or [N, K]")
        if cluster_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("cluster_ids must have an integer dtype")
        if cluster_ids.shape[0] != x.shape[0] or (cluster_ids.ndim == 2 and cluster_ids.shape[1] == 0):
            raise ValueError("cluster_ids must have shape [N] or [N, K] with one ID for each row of x")
        batched_memberships = cluster_ids.ndim == 2
        if batched_memberships:
            if edge_index.shape[0] != x.shape[0]:
                raise ValueError("batched edge_index must contain one row for each x row")
        elif edge_index.shape[0] != 2:
            raise ValueError("edge_index must be a tensor with shape [2, E]")
        if x.device != edge_index.device or x.device != cluster_ids.device:
            raise ValueError("x, edge_index, and cluster_ids must be on the same device")
        if type(cluster_id) is int:
            selected_cluster = cluster_id
        elif isinstance(cluster_id, torch.Tensor) and cluster_id.ndim == 0:
            if cluster_id.dtype not in (torch.int32, torch.int64):
                raise ValueError("cluster_id must have an integer dtype")
            if cluster_id.device != x.device:
                raise ValueError("cluster_id must be on the same device as x")
            selected_cluster = cluster_id
        else:
            raise ValueError("cluster_id must be an integer scalar")

        if batched_memberships:
            return x[(cluster_ids == selected_cluster).any(dim=1)]

        edges = edge_index.to(dtype=torch.long)
        if edges.numel() and bool(((edges < 0) | (edges >= x.shape[0])).any()):
            raise ValueError("edge endpoints must lie in [0, N)")

        selected_nodes = cluster_ids == selected_cluster
        node_indices = torch.nonzero(selected_nodes, as_tuple=True)[0]
        local_index = torch.full((x.shape[0],), -1, dtype=torch.long, device=x.device)
        local_index[node_indices] = torch.arange(node_indices.numel(), device=x.device)
        edge_mask = (local_index[edges[0]] >= 0) & (local_index[edges[1]] >= 0)
        sub_edge_index = local_index[edges[:, edge_mask]]
        features = x[node_indices]
        return features, sub_edge_index, node_indices
