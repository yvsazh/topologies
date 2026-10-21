import torch
from torch import nn

class ProjectionHead(nn.Module):
    component_id = 'projection-head'

    def __init__(self, in_features, out_features, hidden_features=None, normalize=False):
        super().__init__()
        if min(in_features, out_features) <= 0 or (hidden_features is not None and hidden_features <= 0):
            raise ValueError('dimensions must be positive')
        self.net = nn.Linear(in_features, out_features) if hidden_features is None else nn.Sequential(nn.Linear(in_features, hidden_features), nn.GELU(), nn.Linear(hidden_features, out_features))
        self.normalize = normalize

    def forward(self, x):
        y = self.net(x)
        return y / y.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(y.dtype).eps) if self.normalize else y
