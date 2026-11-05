import torch
from torch import nn


class FusionInDecoder(nn.Module):
    component_id = "fid"

    def __init__(self, *args, **kwargs):
        if args or kwargs:
            raise ValueError("FusionInDecoder does not accept constructor arguments")
        super().__init__()

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("encoded_passages must be a torch.Tensor")
        if x.ndim != 4:
            raise ValueError("encoded_passages must have rank 4 with shape (B, P, T, D)")
        if any(size <= 0 for size in x.shape):
            raise ValueError("encoded_passages dimensions must all be positive")
        return x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
