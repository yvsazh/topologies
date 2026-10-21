import torch
from torch import nn
from torch.nn import functional as F


class MusicGen(nn.Module):
    architecture_id = "musicgen"
    def __init__(self, num_codebooks=2, codebook_size=6, hidden_size=8, num_heads=2, max_sequence_length=4096):
        super().__init__()
        if num_codebooks < 1 or codebook_size < 1 or hidden_size < 1 or num_heads < 1:
            raise ValueError("dimensions must be positive")
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if max_sequence_length < 2:
            raise ValueError("max_sequence_length must be at least two")
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_sequence_length = max_sequence_length
        self.token_embeddings = nn.ModuleList(
            nn.Embedding(codebook_size, hidden_size) for _ in range(num_codebooks)
        )
        self.position_embedding = nn.Embedding(max_sequence_length, hidden_size)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.attention_output = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.output_heads = nn.ModuleList(
            nn.Linear(hidden_size, codebook_size) for _ in range(num_codebooks)
        )

    def delayed_embeddings(self, tokens):
        if tokens.ndim != 3 or tokens.shape[-1] != self.num_codebooks:
            raise ValueError("tokens must have shape [batch, time, num_codebooks]")
        batch, time, _ = tokens.shape
        sequence_length = time + self.num_codebooks
        if sequence_length > self.max_sequence_length:
            raise ValueError("delayed sequence exceeds max_sequence_length")
        embeddings = self.token_embeddings[0].weight.new_zeros(
            batch, sequence_length, self.num_codebooks, self.hidden_size
        )
        for index, embedding in enumerate(self.token_embeddings):
            positions = torch.arange(time, device=tokens.device) + index + 1
            embeddings[:, positions, index] = embedding(tokens[..., index])
        return embeddings

    def delayed_targets(self, tokens):
        if tokens.ndim != 3 or tokens.shape[-1] != self.num_codebooks:
            raise ValueError("tokens must have shape [batch, time, num_codebooks]")
        batch, time, _ = tokens.shape
        sequence_length = time + self.num_codebooks - 1
        targets = torch.full(
            (batch, sequence_length, self.num_codebooks), -100,
            dtype=torch.long, device=tokens.device,
        )
        positions = torch.arange(sequence_length, device=tokens.device)
        for index in range(self.num_codebooks):
            frames = positions - index
            valid = (frames >= 0) & (frames < time)
            targets[:, valid, index] = tokens[:, frames[valid], index]
        return targets

    def forward(self, tokens):
        delayed = self.delayed_embeddings(tokens)
        batch, sequence_length, _, _ = delayed.shape
        hidden = delayed.sum(dim=2)
        positions = torch.arange(sequence_length, device=tokens.device)
        hidden = hidden + self.position_embedding(positions).unsqueeze(0)
        shape = (batch, sequence_length, self.num_heads, self.head_dim)
        query = self.query(hidden).view(shape).transpose(1, 2)
        key = self.key(hidden).view(shape).transpose(1, 2)
        value = self.value(hidden).view(shape).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        future_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=tokens.device),
            diagonal=1,
        )
        scores = scores.masked_fill(future_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, value).transpose(1, 2).contiguous().view(batch, sequence_length, self.hidden_size)
        hidden = self.norm1(hidden + self.attention_output(attended))
        hidden = self.norm2(hidden + self.feed_forward(hidden))
        prediction_states = hidden[:, :-1]
        return torch.stack([head(prediction_states) for head in self.output_heads], dim=2)

    def loss(self, tokens):
        logits = self(tokens)
        targets = self.delayed_targets(tokens)
        return F.cross_entropy(
            logits.reshape(-1, self.codebook_size),
            targets.reshape(-1),
            ignore_index=-100,
        )
