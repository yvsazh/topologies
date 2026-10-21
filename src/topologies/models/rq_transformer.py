import torch
from torch import nn


class RQTransformer(nn.Module):
    architecture_id = "rq-transformer"

    def __init__(self, vocab_size=8, depth=2, hidden_dim=8, num_layers=2, num_heads=2):
        super().__init__()
        if vocab_size < 2 or depth < 1 or hidden_dim < 1:
            raise ValueError("vocab_size must be at least 2, depth and hidden_dim must be positive")
        if num_layers < 1 or num_heads < 1 or hidden_dim % num_heads:
            raise ValueError("num_layers and num_heads must be positive and divide hidden_dim")
        self.vocab_size = vocab_size
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.depth_embedding = nn.Embedding(depth, hidden_dim)
        self.position_embedding = nn.Embedding(4096, hidden_dim)
        block = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(block, num_layers=num_layers)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, codes):
        if codes.ndim != 3 or codes.shape[1] != self.depth:
            raise ValueError("codes must have shape (batch, depth, sequence_length)")
        if codes.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError("codes must contain integer token indices")
        if codes.numel() and (codes.min() < 0 or codes.max() >= self.vocab_size):
            raise ValueError("code index is outside the vocabulary")
        batch, _, length = codes.shape
        if length > self.position_embedding.num_embeddings:
            raise ValueError("sequence length exceeds supported positions")
        flat_codes = codes.transpose(1, 2).reshape(batch, length * self.depth)
        shifted = torch.zeros_like(flat_codes)
        shifted[:, 1:] = flat_codes[:, :-1]
        steps = torch.arange(length * self.depth, device=codes.device)
        spatial_positions = steps.div(self.depth, rounding_mode="floor")
        code_depths = steps.remainder(self.depth)
        hidden = self.token_embedding(shifted)
        hidden = hidden + self.position_embedding(spatial_positions)[None]
        hidden = hidden + self.depth_embedding(code_depths)[None]

        size = flat_codes.shape[1]
        causal_mask = torch.triu(
            torch.ones(size, size, device=codes.device, dtype=torch.bool), diagonal=1
        )
        logits = self.output(self.transformer(hidden, mask=causal_mask))
        return logits.reshape(batch, length, self.depth, self.vocab_size).transpose(1, 2)
