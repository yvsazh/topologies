import torch
from torch import nn


class SAGAN(nn.Module):
    architecture_id = "sagan"
    def __init__(self, channels: int, heads: int = 1):
        super().__init__()
        if channels <= 0 or heads <= 0:
            raise ValueError("channels and heads must be positive")
        if channels % heads != 0:
            raise ValueError("channels must be divisible by heads")
        self.channels = channels
        self.heads = heads
        self.query = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.key = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.value = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.output = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(()))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.channels:
            raise ValueError("inputs must have shape (batch, channels, height, width)")
        batch, channels, height, width = inputs.shape
        positions = height * width
        head_channels = channels // self.heads

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(batch, self.heads, head_channels, positions).transpose(-2, -1)

        query = split_heads(self.query(inputs))
        key = split_heads(self.key(inputs))
        value = split_heads(self.value(inputs))
        scores = torch.matmul(query, key.transpose(-2, -1))
        attention = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attention, value).transpose(-2, -1).reshape(batch, channels, height, width)
        return inputs + self.gamma * self.output(attended)
