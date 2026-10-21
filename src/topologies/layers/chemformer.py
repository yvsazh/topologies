import math

import torch
from torch import nn


class Chemformer(nn.Module):
    component_id = "chemformer"
    def __init__(self, vocab_size=32, d_model=32, num_heads=4, num_layers=1, dropout=0.0, max_length=256):
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, d_model)
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        position_encoding = torch.zeros(max_length, d_model)
        position_encoding[:, 0::2] = torch.sin(positions * frequencies)
        position_encoding[:, 1::2] = torch.cos(
            positions * frequencies[: position_encoding[:, 1::2].shape[1]]
        )
        self.register_buffer("position_encoding", position_encoding, persistent=False)
        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model,
                    num_heads,
                    4 * d_model,
                    dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model,
                    num_heads,
                    4 * d_model,
                    dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.output = nn.Linear(d_model, vocab_size)

    def _embed(self, tokens):
        if tokens.ndim != 2 or tokens.shape[1] > self.max_length:
            raise ValueError("token ids must have shape (batch, length) within max_length")
        position_encoding = self.position_encoding[: tokens.shape[1]].to(tokens.device)
        return self.embedding(tokens) * math.sqrt(self.d_model) + position_encoding.unsqueeze(0)

    def encode(self, reactant_ids):
        memory = self._embed(reactant_ids)
        for layer in self.encoder_layers:
            memory = layer(memory)
        return memory

    def decode(self, target_ids, memory):
        target = self._embed(target_ids)
        length = target_ids.shape[1]
        causal_mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=target_ids.device), diagonal=1)
        for layer in self.decoder_layers:
            target = layer(target, memory, tgt_mask=causal_mask)
        return self.output(target)

    def forward(self, reactant_ids, target_ids):
        return self.decode(target_ids, self.encode(reactant_ids))
