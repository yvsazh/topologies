import torch
from torch import nn


class UViTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4):
        super().__init__()
        if hidden_dim <= 0 or num_heads <= 0 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        self.skip_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )

    def forward(self, tokens, skip_source):
        if tokens.shape != skip_source.shape:
            raise ValueError("tokens and skip_source must have identical shapes")
        fused = self.skip_projection(torch.cat((tokens, skip_source), dim=-1))
        normalized = self.norm1(fused)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        hidden = fused + attended
        return hidden + self.mlp(self.norm2(hidden))


class UViT(nn.Module):
    architecture_id = "u-vit"
    def __init__(self, hidden_dim=256, num_heads=8, depth=4, mlp_ratio=4):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least 2 to provide a long skip")
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        self.shallow = nn.ModuleList(
            UViTBlock(hidden_dim, num_heads, mlp_ratio) for _ in range(depth // 2)
        )
        self.deep = nn.ModuleList(
            UViTBlock(hidden_dim, num_heads, mlp_ratio) for _ in range(depth - depth // 2)
        )

    def forward(self, tokens):
        shallow = tokens
        for block in self.shallow:
            shallow = block(shallow, shallow)
        hidden = shallow
        for index, block in enumerate(self.deep):
            hidden = block(hidden, shallow if index == len(self.deep) - 1 else hidden)
        return hidden
