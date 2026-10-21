import math

import torch
from torch import nn


class MaskGIT(nn.Module):
    architecture_id = "maskgit"

    def __init__(self, vocab_size=8, hidden_dim=8, num_tokens=6, num_layers=2, num_heads=2):
        super().__init__()
        if vocab_size <= 1 or hidden_dim <= 0 or num_tokens <= 0:
            raise ValueError("vocab_size, hidden_dim, and num_tokens must be positive")
        if num_layers <= 0 or num_heads <= 0 or hidden_dim % num_heads:
            raise ValueError("num_layers and num_heads must be positive and hidden_dim divisible by num_heads")
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.mask_token_id = vocab_size
        self.token_embedding = nn.Embedding(vocab_size + 1, hidden_dim)
        self.position_embedding = nn.Embedding(num_tokens, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output = nn.Linear(hidden_dim, vocab_size)

    @staticmethod
    def cosine_mask_schedule(step, num_tokens):
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        value = torch.as_tensor(step, dtype=torch.float32).clamp(0.0, 1.0)
        count = torch.ceil(num_tokens * torch.cos(value * (math.pi / 2)))
        return count.to(torch.long)

    def forward(self, tokens, mask=None, step=None):
        if tokens.ndim != 2 or tokens.shape[1] != self.num_tokens:
            raise ValueError("tokens must have shape (batch, num_tokens)")
        if torch.any((tokens < 0) | (tokens >= self.vocab_size)):
            raise ValueError("token IDs must be within the vocabulary")
        if mask is None:
            if step is None:
                raise ValueError("provide mask or step")
            counts = self.cosine_mask_schedule(step, self.num_tokens)
            if counts.ndim == 0:
                counts = counts.expand(tokens.shape[0])
            if counts.numel() != tokens.shape[0]:
                raise ValueError("step must be scalar or have one value per batch item")
            mask = torch.zeros_like(tokens, dtype=torch.bool)
            for row, count in enumerate(counts.tolist()):
                if count:
                    indices = torch.randperm(self.num_tokens, device=tokens.device)[:count]
                    mask[row, indices] = True
        if mask.shape != tokens.shape:
            raise ValueError("mask must match token shape")
        masked = tokens.masked_fill(mask.bool(), self.mask_token_id)
        positions = torch.arange(self.num_tokens, device=tokens.device)
        hidden = self.token_embedding(masked) + self.position_embedding(positions)[None, :, :]
        logits = self.output(self.transformer(hidden))
        return logits

    def predict_masked(self, tokens, mask):
        logits = self.forward(tokens, mask=mask)
        prediction = logits.argmax(dim=-1)
        return torch.where(mask.bool(), prediction, tokens)

    @torch.no_grad()
    def refine(self, tokens, mask, steps=8):
        if steps <= 0:
            raise ValueError("steps must be positive")
        result = tokens.clone()
        masked = mask.bool().clone()
        for index in range(steps):
            if not masked.any():
                break
            logits = self.forward(result, mask=masked)
            probabilities = logits.softmax(dim=-1)
            confidence, prediction = probabilities.max(dim=-1)
            result = torch.where(masked, prediction, result)
            remaining = int(self.cosine_mask_schedule((index + 1) / steps, tokens.shape[1]))
            next_mask = torch.zeros_like(masked)
            for row in range(tokens.shape[0]):
                candidates = torch.where(masked[row])[0]
                keep = min(remaining, candidates.numel())
                if keep:
                    chosen = candidates[confidence[row, candidates].argsort()[:keep]]
                    next_mask[row, chosen] = True
            masked = next_mask
        return result
