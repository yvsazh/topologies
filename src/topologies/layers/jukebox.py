import torch
from torch import nn
from torch.nn import functional as F


class Jukebox(nn.Module):
    architecture_id = "jukebox"
    component_id = "jukebox"

    def __init__(self, coarse_vocab_size, fine_vocab_size, hidden_size, num_heads=2):
        super().__init__()
        if min(coarse_vocab_size, fine_vocab_size, hidden_size, num_heads) <= 0:
            raise ValueError("vocabulary sizes, hidden_size, and num_heads must be positive")
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.coarse_embedding = nn.Embedding(coarse_vocab_size, hidden_size)
        self.fine_embedding = nn.Embedding(fine_vocab_size, hidden_size)
        self.prior = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=4 * hidden_size,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.fine_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, fine_vocab_size),
        )

    def forward(self, coarse_tokens, fine_tokens):
        if coarse_tokens.ndim != 2 or fine_tokens.ndim != 2:
            raise ValueError("coarse_tokens and fine_tokens must have shape [batch, length]")
        if coarse_tokens.shape[0] != fine_tokens.shape[0]:
            raise ValueError("coarse_tokens and fine_tokens must have matching batch sizes")
        if coarse_tokens.shape[1] == 0 or fine_tokens.shape[1] == 0:
            raise ValueError("coarse_tokens and fine_tokens must be nonempty")
        coarse = self.coarse_embedding(coarse_tokens)
        coarse = F.interpolate(
            coarse.transpose(1, 2), size=fine_tokens.shape[1], mode="linear", align_corners=False
        ).transpose(1, 2)
        conditioned = coarse + self.fine_embedding(fine_tokens)
        length = fine_tokens.shape[1]
        causal_mask = torch.triu(torch.ones(length, length, dtype=torch.bool,
                                            device=fine_tokens.device), diagonal=1)
        hidden = self.prior(conditioned, src_mask=causal_mask)
        return self.fine_predictor(hidden)
