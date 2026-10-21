import torch
from torch import nn

def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')

class GraphConvolution(nn.Module):
    component_id = 'graph-conv'

    def __init__(self, in_features: int, out_features: int, mode: str='gcn', self_loops: bool=True) -> None:
        super().__init__()
        _positive_int(in_features, 'in_features')
        _positive_int(out_features, 'out_features')
        if mode not in ('gcn', 'mean_graphsage') or not isinstance(self_loops, bool):
            raise ValueError('mode must be gcn or mean_graphsage and self_loops must be bool')
        self.linear = nn.Linear(in_features * (2 if mode == 'mean_graphsage' else 1), out_features, bias=mode == 'mean_graphsage')
        self.mode, self.self_loops, self.in_features = (mode, self_loops, in_features)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None=None) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.in_features or (not x.is_floating_point()):
            raise ValueError('x must be floating [N,in_features]')
        n = x.shape[0]
        if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype not in (torch.int32, torch.int64):
            raise ValueError('edge_index must be an integer tensor of shape [2,E]')
        if edge_index.device != x.device:
            raise ValueError('edge_index and x must be on the same device')
        src, dst = edge_index.long()
        if src.numel() and bool(((src < 0) | (src >= n) | (dst < 0) | (dst >= n)).any()):
            raise ValueError('edge indices must lie in [0,N)')
        if edge_weight is None:
            edge_weight = x.new_ones(src.shape)
        elif edge_weight.ndim != 1 or edge_weight.numel() != src.numel() or edge_weight.device != x.device or (not edge_weight.is_floating_point()):
            raise ValueError("edge_weight must be floating [E] on x's device")
        else:
            edge_weight = edge_weight.to(dtype=x.dtype)
        if not bool(torch.isfinite(edge_weight).all()) or bool((edge_weight < 0).any()):
            raise ValueError('edge weights must be finite and nonnegative')
        if self.self_loops:
            has_loop = torch.zeros(n, dtype=torch.bool, device=x.device)
            loop_mask = src == dst
            if bool(loop_mask.any()):
                has_loop[src[loop_mask]] = True
            missing = torch.arange(n, device=x.device)[~has_loop]
            src = torch.cat((src, missing))
            dst = torch.cat((dst, missing))
            edge_weight = torch.cat((edge_weight, x.new_ones(missing.shape)))
        if self.mode == 'mean_graphsage':
            neighbor = src != dst
            src, dst, edge_weight = (src[neighbor], dst[neighbor], edge_weight[neighbor])
        aggregate = x.new_zeros(x.shape)
        degree = x.new_zeros(n)
        aggregate.index_add_(0, dst, x[src] * edge_weight[:, None])
        degree.index_add_(0, dst, edge_weight)
        if self.mode == 'mean_graphsage':
            denominator = torch.where(degree > 0, degree, torch.ones_like(degree))
            neighbor_mean = aggregate / denominator[:, None]
            return self.linear(torch.cat((x, neighbor_mean), dim=-1))
        safe_degree = torch.where(degree > 0, degree, torch.ones_like(degree))
        inv_degree = safe_degree.rsqrt()
        normalized = edge_weight * inv_degree[src] * inv_degree[dst]
        aggregate = x.new_zeros(x.shape)
        aggregate.index_add_(0, dst, x[src] * normalized[:, None])
        return self.linear(aggregate)
