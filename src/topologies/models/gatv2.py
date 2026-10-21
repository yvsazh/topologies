import torch
from torch import nn
from torch.nn import functional as F


class GATv2(nn.Module):
    architecture_id = "gatv2"
    def __init__(self, in_features, out_features, heads=1, concat=True, negative_slope=0.2, dropout=0.0, bias=True):
        super().__init__()
        if in_features <= 0 or out_features <= 0 or heads <= 0:
            raise ValueError("feature sizes and heads must be positive")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.projection = nn.Linear(in_features, heads * out_features, bias=False)
        self.attention = nn.Parameter(torch.empty(heads, out_features))
        self.bias = nn.Parameter(torch.zeros(heads * out_features if concat else out_features)) if bias else None
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.xavier_uniform_(self.attention)

    def forward(self, x, edge_index):
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError("x must have shape (nodes, in_features)")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, edges)")
        edge_index = edge_index.to(device=x.device, dtype=torch.long)
        source, target = edge_index
        if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= x.shape[0]):
            raise ValueError("edge_index contains an invalid node")
        projected = self.projection(x).view(x.shape[0], self.heads, self.out_features)
        pair = projected[source] + projected[target]
        logits = (F.leaky_relu(pair, self.negative_slope) * self.attention).sum(-1)
        edge_weights = torch.zeros_like(logits)
        for node in range(x.shape[0]):
            selected = target == node
            if selected.any():
                edge_weights[selected] = torch.softmax(logits[selected], dim=0)
        edge_weights = F.dropout(edge_weights, p=self.dropout, training=self.training)
        messages = projected[source] * edge_weights.unsqueeze(-1)
        output = x.new_zeros((x.shape[0], self.heads, self.out_features))
        output.index_add_(0, target, messages)
        output = output.reshape(x.shape[0], -1) if self.concat else output.mean(dim=1)
        if self.bias is not None:
            output = output + self.bias
        return output
