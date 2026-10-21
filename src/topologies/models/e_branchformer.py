import torch
from torch import nn
from torch.nn import functional as F


class EBranchformer(nn.Module):
    architecture_id = "e-branchformer"
    def __init__(self, channels=8, num_heads=2, kernel_size=3, ff_multiplier=4, dropout=0.0):
        super().__init__()
        if channels <= 0 or num_heads <= 0 or channels % num_heads:
            raise ValueError("channels must be positive and divisible by num_heads")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if ff_multiplier <= 0 or not 0.0 <= dropout < 1.0:
            raise ValueError("ff_multiplier must be positive and dropout must be in [0, 1)")
        self.channels = channels
        self.local_norm = nn.LayerNorm(channels)
        self.local_in = nn.Linear(channels, channels * 2)
        self.local_gate_norm = nn.LayerNorm(channels)
        self.local_conv = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels)
        self.local_out = nn.Linear(channels, channels)
        self.attn_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.merge_depthwise = nn.Conv1d(channels * 2, channels * 2, kernel_size,
                                         padding=kernel_size // 2, groups=channels * 2)
        self.merge = nn.Linear(channels * 2, channels)
        self.merge_norm = nn.LayerNorm(channels)
        hidden = channels * ff_multiplier
        self.ff_in = nn.Linear(channels, hidden)
        self.ff_out = nn.Linear(hidden, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.channels:
            raise ValueError("input must have shape (batch, sequence, channels)")
        projected = F.gelu(self.local_in(self.local_norm(x)))
        content, gate = projected.chunk(2, dim=-1)
        gate = self.local_gate_norm(gate)
        gate = self.local_conv(gate.transpose(1, 2)).transpose(1, 2)
        local = self.local_out(content * gate)
        normalized = self.attn_norm(x)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        merged = self.merge_branches(attended, local)
        x = x + self.dropout(merged)
        ff = self.ff_out(self.dropout(F.gelu(self.ff_in(self.merge_norm(x)))))
        return x + 0.5 * self.dropout(ff)

    def merge_branches(self, global_branch, local_branch):
        concatenated = torch.cat((global_branch, local_branch), dim=-1)
        temporal = self.merge_depthwise(concatenated.transpose(1, 2)).transpose(1, 2)
        return self.merge(concatenated + temporal)
