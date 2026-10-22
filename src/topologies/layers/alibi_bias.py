import torch
from torch import nn


class ALiBiBias(nn.Module):
    component_id = "alibi-bias"

    def __init__(self, num_heads=8):
        super().__init__()
        if isinstance(num_heads, bool) or not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError("num_heads must be a positive integer")
        ratio = 2.0 ** (-8.0 / num_heads)
        exponents = torch.arange(1, num_heads + 1, dtype=torch.float32)
        self.register_buffer("slopes", ratio**exponents)

    def forward(self, scores):
        if not isinstance(scores, torch.Tensor) or scores.ndim != 4:
            raise ValueError("attention scores must be a tensor with 4 dimensions")
        if not scores.is_floating_point():
            raise ValueError("attention scores must have a floating-point dtype")
        num_heads, query_length, key_length = scores.shape[1:]
        if num_heads != self.slopes.numel():
            raise ValueError(f"attention scores must have num_heads={self.slopes.numel()}")
        if query_length != key_length:
            raise ValueError("attention scores must have equal query and key lengths")
        positions = torch.arange(query_length, device=scores.device)
        distances = (positions[:, None] - positions[None, :]).abs().to(dtype=scores.dtype)
        slopes = self.slopes.to(device=scores.device, dtype=scores.dtype)
        bias = -slopes[:, None, None] * distances[None, :, :]
        return scores + bias.unsqueeze(0)
