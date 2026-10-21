import torch
from torch import nn

class GraphReadout(nn.Module):
    component_id = 'graph-readout'

    def __init__(self, mode='mean', empty_value=0.0):
        super().__init__()
        if mode not in ('sum', 'mean', 'max'):
            raise ValueError('invalid readout mode')
        self.mode = mode
        self.empty_value = empty_value

    def forward(self, x, batch, num_graphs=None):
        if x.ndim < 2 or batch.ndim != 1 or batch.numel() != x.shape[0] or (batch.dtype not in (torch.int32, torch.int64)) or (batch.device != x.device):
            raise ValueError("batch must be an integer [N] graph-id tensor on x's device")
        if num_graphs is not None and (isinstance(num_graphs, bool) or not isinstance(num_graphs, int) or num_graphs < 0):
            raise ValueError('num_graphs must be a nonnegative integer')
        n = int(batch.max().item()) + 1 if batch.numel() and num_graphs is None else num_graphs or 0
        if batch.numel() and bool(((batch < 0) | (batch >= n)).any()):
            raise ValueError('invalid graph index')
        if self.mode == 'max':
            out = x.new_full((n, *x.shape[1:]), -torch.inf)
            out.scatter_reduce_(0, batch.reshape(-1, *[1] * (x.ndim - 1)).expand_as(x), x, reduce='amax', include_self=True)
            return torch.where(torch.isfinite(out), out, out.new_full((), self.empty_value))
        out = x.new_zeros((n, *x.shape[1:]))
        out.index_add_(0, batch, x)
        if self.mode == 'mean':
            count = x.new_zeros(n)
            count.index_add_(0, batch, x.new_ones(batch.shape))
            out = out / count.clamp_min(1).reshape(n, *[1] * (x.ndim - 1))
        return out
