import torch
from torch import nn
from torch.nn import functional as F


class SentenceSimilarity(nn.Module):
    component_id = "sentence-bert"

    def forward(self, left, right, left_mask, right_mask):
        if left.ndim != 3 or right.ndim != 3:
            raise ValueError("left and right must have shape (batch, tokens, features)")
        if left.shape[0] != right.shape[0] or left.shape[2] != right.shape[2]:
            raise ValueError("sentence batches and feature dimensions must match")
        if left_mask.shape != left.shape[:2] or right_mask.shape != right.shape[:2]:
            raise ValueError("each mask must match its sentence batch and token dimensions")

        left_weights = left_mask.to(device=left.device).bool().to(dtype=left.dtype).unsqueeze(-1)
        right_weights = right_mask.to(device=right.device).bool().to(dtype=right.dtype).unsqueeze(-1)
        left_pool = (left * left_weights).sum(dim=1) / left_weights.sum(dim=1).clamp_min(1)
        right_pool = (right * right_weights).sum(dim=1) / right_weights.sum(dim=1).clamp_min(1)
        return (F.normalize(left_pool, p=2, dim=-1) * F.normalize(right_pool, p=2, dim=-1)).sum(dim=-1)
