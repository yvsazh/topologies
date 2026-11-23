import torch
from torch import nn


class SynthesizerAttention(nn.Module):
    component_id = "synthesizer"

    def __init__(self, d_model, max_len):
        super().__init__()
        if type(d_model) is not int or d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        if type(max_len) is not int or max_len <= 0:
            raise ValueError("max_len must be a positive integer")
        self.d_model = d_model
        self.max_len = max_len
        self.W1 = nn.Linear(d_model, d_model, bias=False)
        self.W2 = nn.Linear(d_model, max_len, bias=False)

    def forward(self, x, values):
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError("x must be a rank-3 tensor shaped [batch, length, d_model]")
        if not isinstance(values, torch.Tensor) or values.ndim != 3:
            raise ValueError("values must be a rank-3 tensor shaped [batch, length, d_model]")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"x feature dimension must equal d_model ({self.d_model})")
        if values.shape != x.shape:
            raise ValueError("values must have the same shape as x")
        length = x.shape[1]
        if length == 0:
            raise ValueError("sequence length must be positive")
        if length > self.max_len:
            raise ValueError(f"sequence length cannot exceed max_len ({self.max_len})")
        logits = self.W2(torch.relu(self.W1(x)))[:, :, :length]
        attention = torch.softmax(logits, dim=-1)
        return torch.matmul(attention, values)
