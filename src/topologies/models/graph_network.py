import torch
from torch import nn
from torch.nn import functional as F

class GraphNetwork(nn.Module):
    architecture_id = 'graph-network'

    def __init__(self, in_features, hidden_features, num_classes, layers=2, mode='gcn', readout='mean', dropout=0.0):
        super().__init__()
        if min(in_features, hidden_features, num_classes, layers) < 1:
            raise ValueError('feature, class, and layer counts must be positive')
        if mode not in {'gcn', 'mean_graphsage'} or readout not in {'mean', 'sum', 'max'}:
            raise ValueError('unsupported graph mode or readout')
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0, 1)')
        self.mode, self.readout, self.dropout = (mode, readout, dropout)
        widths = [in_features] + [hidden_features] * layers
        self.layers = nn.ModuleList((nn.Linear(widths[i] * (2 if mode == 'mean_graphsage' else 1), widths[i + 1]) for i in range(layers)))
        self.classifier = nn.Linear(hidden_features, num_classes)

    def forward(self, x, edge_index, edge_weight=None, batch=None, node_level=True):
        if x.ndim != 2 or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError('x must be [N,F] and edge_index [2,E]')
        n = x.shape[0]
        src, dst = edge_index.long()
        if edge_weight is None:
            edge_weight = x.new_ones(src.shape[0])
        if edge_weight.shape != src.shape:
            raise ValueError('edge_weight must have shape [E]')
        h = x
        for layer in self.layers:
            if self.mode == 'gcn':
                all_src = torch.cat((src, torch.arange(n, device=x.device)))
                all_dst = torch.cat((dst, torch.arange(n, device=x.device)))
                weights = torch.cat((edge_weight.to(x), x.new_ones(n)))
                degree = x.new_zeros(n).index_add_(0, all_dst, weights)
                norm = weights * degree[all_src].clamp_min(1).rsqrt() * degree[all_dst].clamp_min(1).rsqrt()
                agg = x.new_zeros((n, h.shape[1])).index_add_(0, all_dst, h[all_src] * norm[:, None])
                h = layer(agg)
            else:
                agg = h.new_zeros(h.shape).index_add_(0, dst, h[src] * edge_weight.to(h)[:, None])
                degree = h.new_zeros(n).index_add_(0, dst, edge_weight.to(h)).clamp_min(1)
                h = layer(torch.cat((h, agg / degree[:, None]), dim=-1))
            if layer is not self.layers[-1]:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        if node_level:
            return self.classifier(h)
        if batch is None or batch.shape != (n,):
            raise ValueError('graph-level output requires batch[N]')
        count = int(batch.max().item()) + 1 if batch.numel() else 0
        pooled = h.new_zeros((count, h.shape[1]))
        if self.readout == 'max':
            pooled.fill_(float('-inf'))
            pooled.scatter_reduce_(0, batch[:, None].expand_as(h), h, reduce='amax', include_self=True)
            pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))
        else:
            pooled.index_add_(0, batch, h)
            if self.readout == 'mean':
                pooled /= torch.bincount(batch, minlength=count).clamp_min(1).to(h)[:, None]
        return self.classifier(pooled)
