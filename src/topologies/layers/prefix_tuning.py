import torch
from torch import nn


class PrefixTuning(nn.Module):
    component_id = "prefix-tuning"

    def __init__(self, num_heads, head_dim, prefix_len=10):
        super().__init__()
        for name, value in (("num_heads", num_heads), ("head_dim", head_dim), ("prefix_len", prefix_len)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.prefix_len = prefix_len
        self.prefix_keys = nn.Parameter(torch.empty(num_heads, prefix_len, head_dim))
        self.prefix_values = nn.Parameter(torch.empty(num_heads, prefix_len, head_dim))
        nn.init.uniform_(self.prefix_keys, 0.0, 1.0)
        nn.init.uniform_(self.prefix_values, 0.0, 1.0)

    def forward(self, keys, values):
        if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
            raise TypeError("keys and values must be torch.Tensor instances")
        if keys.ndim != 4 or values.ndim != 4:
            raise ValueError("keys and values must have rank 4 with shape (batch, heads, sequence, head_dim)")
        if keys.shape != values.shape:
            raise ValueError("keys and values must have identical shapes")
        if any(size <= 0 for size in keys.shape):
            raise ValueError("keys and values dimensions must be positive")
        if keys.shape[1] != self.num_heads:
            raise ValueError(f"keys and values num_heads must match configured num_heads {self.num_heads}")
        if keys.shape[3] != self.head_dim:
            raise ValueError(f"keys and values head_dim must match configured head_dim {self.head_dim}")
        if not keys.is_floating_point() or not values.is_floating_point():
            raise ValueError("keys and values must use floating-point dtypes")
        if keys.device != values.device or keys.dtype != values.dtype:
            raise ValueError("keys and values must have matching dtype and device")
        if keys.device != self.prefix_keys.device or keys.dtype != self.prefix_keys.dtype:
            raise ValueError("keys and values must match the prefix parameters' dtype and device")
        batch_size = keys.shape[0]
        prefix_keys = self.prefix_keys.unsqueeze(0).expand(batch_size, -1, -1, -1)
        prefix_values = self.prefix_values.unsqueeze(0).expand(batch_size, -1, -1, -1)
        output_keys = torch.cat((prefix_keys, keys), dim=2)
        output_values = torch.cat((prefix_values, values), dim=2)
        return torch.stack((output_keys, output_values), dim=3)
