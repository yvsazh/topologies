import math

import torch
from torch import nn


class MAGVIT(nn.Module):
    architecture_id = "magvit"
    def __init__(self, vocab_size=8, hidden_dim=8, frames=2, tokens=4, num_heads=2, mask_token_id=None):
        super().__init__()
        if vocab_size < 2 or hidden_dim < 1 or frames < 1 or tokens < 1 or num_heads < 1:
            raise ValueError("vocab_size, hidden_dim, frames, and tokens must be positive")
        if mask_token_id is not None and mask_token_id != vocab_size:
            raise ValueError("mask_token_id must equal vocab_size")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.vocab_size = vocab_size
        self.frames = frames
        self.tokens = tokens
        self.mask_token_id = vocab_size if mask_token_id is None else mask_token_id
        self.token_embedding = nn.Embedding(vocab_size + 1, hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, frames * tokens, hidden_dim))
        layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 2, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.output = nn.Linear(hidden_dim, vocab_size)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, video_tokens, mask=None):
        if video_tokens.ndim != 3:
            raise ValueError("video_tokens must have shape (batch, frames, tokens)")
        batch, frames, tokens = video_tokens.shape
        if frames != self.frames or tokens != self.tokens:
            raise ValueError("video-token volume does not match configured frames and tokens")
        if mask is None:
            mask = video_tokens.eq(self.mask_token_id)
        if mask.shape != video_tokens.shape:
            raise ValueError("mask must have the same shape as video_tokens")
        if torch.any((video_tokens < 0) | (video_tokens >= self.vocab_size)):
            invalid = (video_tokens != self.mask_token_id) & ((video_tokens < 0) | (video_tokens >= self.vocab_size))
            if torch.any(invalid):
                raise ValueError("video tokens must be vocabulary indices or mask_token_id")
        flat_tokens = torch.where(mask, self.mask_token_id, video_tokens).reshape(batch, frames * tokens)
        hidden = self.token_embedding(flat_tokens) + self.position_embedding
        logits = self.output(self.encoder(hidden)).reshape(batch, frames, tokens, self.vocab_size)
        predictions = logits.argmax(dim=-1)
        result = torch.where(mask, predictions, video_tokens)
        return {"logits": logits, "tokens": result, "mask": mask}

    @staticmethod
    def mask_count(total_tokens, step, total_steps, initial_ratio=1.0):
        if total_tokens < 0 or total_steps < 1:
            raise ValueError("total_tokens must be nonnegative and total_steps positive")
        if not 0 <= step <= total_steps or not 0.0 <= initial_ratio <= 1.0:
            raise ValueError("step and initial_ratio are out of range")
        if step == total_steps:
            return 0
        remaining = max(0.0, math.cos(math.pi * step / (2.0 * total_steps)))
        return min(total_tokens, math.ceil(total_tokens * initial_ratio * remaining))

    def refine(self, video_tokens, steps=4, initial_ratio=1.0):
        if steps < 1:
            raise ValueError("steps must be positive")
        current = video_tokens.clone()
        batch = current.shape[0]
        total = self.frames * self.tokens
        editable = current.eq(self.mask_token_id).flatten(1)
        original_mask_count = editable.sum(dim=1)
        active = editable.clone()
        for step in range(steps):
            masked = active.reshape_as(current)
            result = self(current, masked)
            probabilities = result["logits"].softmax(dim=-1)
            confidence, predictions = probabilities.max(dim=-1)
            current = torch.where(masked, predictions, current)
            if step + 1 == steps:
                break
            next_active = torch.zeros_like(active)
            for batch_index in range(batch):
                count = min(int(original_mask_count[batch_index].item()),
                            math.ceil(int(original_mask_count[batch_index].item()) *
                                      initial_ratio * math.cos(math.pi * (step + 1) / (2.0 * steps))))
                candidate_positions = editable[batch_index].nonzero(as_tuple=False).squeeze(-1)
                if count and candidate_positions.numel():
                    candidate_confidence = confidence[batch_index].flatten().index_select(0, candidate_positions)
                    selected = candidate_positions[candidate_confidence.argsort()[:count]]
                    next_active[batch_index, selected] = True
                    current.reshape(batch, total)[batch_index, selected] = self.mask_token_id
            active = next_active
        return current
