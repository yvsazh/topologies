import torch
from torch import nn


class ImageGPT(nn.Module):
    architecture_id = "imagegpt"
    def __init__(self, vocab_size=512, hidden_dim=128, num_heads=4, seq_len=256, num_layers=4, mlp_dim=None, dropout=0.0):
        super().__init__()
        if vocab_size < 1 or hidden_dim < 1 or seq_len < 1:
            raise ValueError("vocab_size, hidden_dim, and seq_len must be positive")
        if num_heads < 1 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(seq_len, hidden_dim)
        width = mlp_dim or 4 * hidden_dim
        self.blocks = nn.ModuleList([_CausalBlock(hidden_dim, num_heads, width, dropout) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, tokens):
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (batch, sequence)")
        batch, length = tokens.shape
        if length > self.seq_len:
            raise ValueError("input sequence exceeds configured seq_len")
        if tokens.numel() and (tokens.min() < 0 or tokens.max() >= self.vocab_size):
            raise ValueError("token ids must be within the configured vocabulary")
        positions = torch.arange(length, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions).unsqueeze(0)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(self.final_norm(hidden))


class _CausalBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_dim, dropout):
        super().__init__()
        self.norm_attention = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, hidden_dim), nn.Dropout(dropout))

    def forward(self, hidden):
        length = hidden.shape[1]
        mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=hidden.device), diagonal=1)
        normalized = self.norm_attention(hidden)
        attended = self.attention(normalized, normalized, normalized, attn_mask=mask, need_weights=False)[0]
        hidden = hidden + attended
        return hidden + self.mlp(self.norm_mlp(hidden))
