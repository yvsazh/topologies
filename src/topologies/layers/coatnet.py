import torch
from torch import nn


class CoAtNetRelativeAttentionBias(nn.Module):
    component_id = "coatnet"

    def __init__(self, height, width, num_heads):
        super().__init__()
        for name, value in (("height", height), ("width", width), ("num_heads", num_heads)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.height = height
        self.width = width
        self.num_heads = num_heads
        self.token_count = height * width
        self.bias_table = nn.Parameter(torch.zeros(num_heads, 2 * height - 1, 2 * width - 1))
        positions = torch.arange(self.token_count)
        rows = positions // width
        columns = positions % width
        row_offsets = rows[:, None] - rows[None, :] + height - 1
        column_offsets = columns[:, None] - columns[None, :] + width - 1
        indices = row_offsets * (2 * width - 1) + column_offsets
        self.register_buffer("relative_indices", indices, persistent=False)

    def forward(self, logits):
        if not isinstance(logits, torch.Tensor):
            raise TypeError("logits must be a torch.Tensor")
        if logits.ndim != 4:
            raise ValueError("logits must have shape [batch, num_heads, height*width, height*width]")
        expected_shape = (self.num_heads, self.token_count, self.token_count)
        if logits.shape[0] < 1 or tuple(logits.shape[1:]) != expected_shape:
            raise ValueError(
                "logits must have shape "
                f"[batch, {self.num_heads}, {self.token_count}, {self.token_count}]"
            )
        if not logits.is_floating_point():
            raise ValueError("logits must have a floating-point dtype")
        if logits.device != self.bias_table.device:
            raise ValueError("logits and bias table must be on the same device")
        flat_bias = self.bias_table.reshape(self.num_heads, -1)
        gathered = flat_bias.index_select(1, self.relative_indices.reshape(-1))
        bias = gathered.reshape(1, self.num_heads, self.token_count, self.token_count)
        return logits + bias.to(dtype=logits.dtype)
