import torch
from torch import nn

def _valid_mask(x, mask, lengths):
    batch, time, _ = x.shape
    if mask is not None and lengths is not None:
        raise ValueError('supply mask or lengths, not both')
    if lengths is not None:
        if lengths.ndim != 1 or lengths.numel() != batch or lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError('lengths must be an integer [B] tensor')
        if lengths.device != x.device or bool(((lengths < 1) | (lengths > time)).any()):
            raise ValueError("lengths must be on x's device and lie in [1,T]")
        return torch.arange(time, device=x.device)[None, :] < lengths[:, None]
    if mask is None:
        return torch.ones((batch, time), device=x.device, dtype=torch.bool)
    if mask.shape != (batch, time) or mask.device != x.device:
        raise ValueError("mask must have shape [B,T] on x's device")
    if mask.dtype == torch.bool:
        return mask
    if mask.is_floating_point() or mask.is_complex() or bool(((mask != 0) & (mask != 1)).any()):
        raise ValueError('mask values must be exactly 0/1 or bool (True means valid)')
    return mask == 1

class MaskedPooling(nn.Module):
    component_id = 'masked-pooling'

    def __init__(self, mode='mean'):
        super().__init__()
        if mode not in ('mean', 'max', 'last'):
            raise ValueError('mode must be mean, max or last')
        self.mode = mode

    def forward(self, x, mask=None, lengths=None):
        if x.ndim != 3:
            raise ValueError('expected [B,T,D]')
        valid = _valid_mask(x, mask, lengths)
        if bool((~valid.any(1)).any()):
            raise ValueError('empty sequences are not poolable')
        if self.mode == 'mean':
            return (x * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)
        if self.mode == 'max':
            return x.masked_fill(~valid.unsqueeze(-1), -torch.inf).max(1).values
        idx = valid.sum(1) - 1
        return x[torch.arange(x.shape[0], device=x.device), idx]
