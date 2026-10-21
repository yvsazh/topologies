import torch
from torch import nn
import torch.nn.functional as F

class FIRFilter(nn.Module):
    component_id = 'fir-filter'

    def __init__(self, taps, initial=None):
        super().__init__()
        if taps <= 0:
            raise ValueError('taps must be positive')
        weight = torch.zeros(taps) if initial is None else torch.as_tensor(initial, dtype=torch.float32)
        if weight.numel() != taps:
            raise ValueError('initial tap count mismatch')
        self.taps = nn.Parameter(weight)

    def forward(self, x):
        if x.ndim == 2:
            return F.conv1d(F.pad(x[:, None, :], (self.taps.numel() - 1, 0)), self.taps.flip(0).view(1, 1, -1)).squeeze(1)
        if x.ndim == 3:
            return F.conv1d(F.pad(x, (self.taps.numel() - 1, 0)), self.taps.flip(0).view(1, 1, -1).expand(x.shape[1], 1, -1), groups=x.shape[1])
        raise ValueError('expected [B,T] or [B,C,T]')
