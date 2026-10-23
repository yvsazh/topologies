import math
import numbers

import torch
from torch import nn


def _value_error(message):
    return ValueError(message)


class FedProx(nn.Module):
    component_id = "fedprox"

    def __init__(self, mu, batch_dim=None):
        super().__init__()
        if isinstance(mu, bool) or not isinstance(mu, numbers.Real):
            raise _value_error("mu must be a finite real scalar greater than or equal to zero")
        try:
            mu = float(mu)
        except OverflowError:
            raise _value_error("mu must be a finite real scalar greater than or equal to zero") from None
        if not math.isfinite(mu) or mu < 0:
            raise _value_error("mu must be a finite real scalar greater than or equal to zero")
        if batch_dim is not None and (isinstance(batch_dim, bool) or not isinstance(batch_dim, numbers.Integral)):
            raise _value_error("batch_dim must be an integer axis or None")
        self.mu = mu
        self.batch_dim = None if batch_dim is None else int(batch_dim)

    def forward(self, local_params, global_params):
        if not isinstance(local_params, torch.Tensor) or not isinstance(global_params, torch.Tensor):
            raise _value_error("local_params and global_params must be tensors")
        if local_params.shape != global_params.shape:
            raise _value_error("local_params and global_params must have the same shape")
        delta = local_params - global_params.detach()
        squared_delta = delta.square()
        if self.batch_dim is None:
            return (self.mu / 2) * squared_delta.sum()
        if delta.ndim == 0 or not -delta.ndim <= self.batch_dim < delta.ndim:
            raise _value_error("batch_dim must identify an axis in local_params")
        batched_delta = squared_delta.movedim(self.batch_dim, 0)
        if batched_delta.ndim == 1:
            return (self.mu / 2) * batched_delta
        return (self.mu / 2) * batched_delta.flatten(start_dim=1).sum(dim=1)
