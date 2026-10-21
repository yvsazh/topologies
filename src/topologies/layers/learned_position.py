import torch
from torch import nn


class LearnedPosition(nn.Module):
    component_id = "learned-position"
    def __init__(self, max_len=8, embed_dim=4):
        super().__init__()
        if max_len <= 0 or embed_dim <= 0:
            raise ValueError("max_len and embed_dim must be positive")
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.position_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.shape[-1] != self.embed_dim:
            raise ValueError("tokens must have shape (batch, sequence, embed_dim)")
        sequence_length = tokens.shape[1]
        if sequence_length > self.max_len:
            raise ValueError("sequence length exceeds max_len")
        positions = torch.arange(sequence_length, device=tokens.device)
        return tokens + self.position_embedding(positions).unsqueeze(0)
