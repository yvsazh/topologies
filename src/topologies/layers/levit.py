import math
import numbers

import torch
from torch import nn


class LeViT(nn.Module):
    component_id = "levit"

    def __init__(self, height, width, num_heads, bias_init=0.0):
        super().__init__()
        for name, value in (
            ("height", height),
            ("width", width),
            ("num_heads", num_heads),
        ):
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(bias_init, bool)
            or not isinstance(bias_init, numbers.Real)
            or not math.isfinite(float(bias_init))
        ):
            raise ValueError("bias_init must be a finite real number")
        self.height = int(height)
        self.width = int(width)
        self.num_heads = int(num_heads)
        self.bias_table = nn.Parameter(
            torch.full((self.num_heads, self.height, self.width), float(bias_init))
        )

    def forward(self, logits):
        if not torch.is_tensor(logits) or logits.ndim != 4:
            raise ValueError("logits must be a rank-4 tensor with shape [batch, heads, H*W, H*W]")
        if not logits.is_floating_point():
            raise ValueError("logits must have a floating-point dtype")
        positions = self.height * self.width
        if logits.shape[1] != self.num_heads:
            raise ValueError(f"logits must have {self.num_heads} heads")
        if logits.shape[2:] != (positions, positions):
            raise ValueError(
                f"logits query and key lengths must both equal height*width ({positions})"
            )
        indices = torch.arange(positions, device=logits.device)
        rows = torch.div(indices, self.width, rounding_mode="floor")
        columns = indices.remainder(self.width)
        row_offsets = (rows[:, None] - rows[None, :]).abs()
        column_offsets = (columns[:, None] - columns[None, :]).abs()
        bias = self.bias_table[:, row_offsets, column_offsets].to(
            device=logits.device, dtype=logits.dtype
        )
        return logits + bias.unsqueeze(0)
