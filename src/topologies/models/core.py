import math

import torch
from torch import nn
from torch.nn import functional as F


class CORE(nn.Module):
    architecture_id = "core"
    PRESETS = {
        "tiny": {"num_items": 128, "embedding_dim": 16, "tau": 0.1, "rho": 0.0},
        "small": {"num_items": 10000, "embedding_dim": 64, "tau": 0.1, "rho": 0.0},
        "base": {"num_items": 59593, "embedding_dim": 100, "tau": 0.1, "rho": 0.0},
    }

    def __init__(self, num_items, embedding_dim=100, tau=None, rho=None):
        super().__init__()
        if isinstance(num_items, bool) or not isinstance(num_items, int) or num_items <= 0:
            raise ValueError("num_items must be a positive integer")
        if isinstance(embedding_dim, bool) or not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive integer")
        try:
            if isinstance(tau, bool) or not isinstance(tau, (int, float)):
                raise TypeError
            tau = float(tau)
        except (OverflowError, TypeError, ValueError):
            raise ValueError("tau must be a finite positive number") from None
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("tau must be a finite positive number")
        try:
            if isinstance(rho, bool) or not isinstance(rho, (int, float)):
                raise TypeError
            rho = float(rho)
        except (OverflowError, TypeError, ValueError):
            raise ValueError("rho must be a finite number in [0, 1)") from None
        if not math.isfinite(rho) or not 0 <= rho < 1:
            raise ValueError("rho must be a finite number in [0, 1)")
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.tau = tau
        self.rho = rho
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

    def _validate_history(self, history_ids, history_mask):
        if not isinstance(history_ids, torch.Tensor) or history_ids.ndim != 2:
            raise ValueError("history_ids must be a rank-two tensor with shape [batch, length]")
        if history_ids.shape[0] <= 0 or history_ids.shape[1] <= 0:
            raise ValueError("history_ids must have positive batch and history dimensions")
        if history_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("history_ids must have an integer dtype")
        if history_ids.device != self.item_embeddings.weight.device:
            raise ValueError("history_ids and model parameters must be on the same device")
        if torch.any(history_ids < 0) or torch.any(history_ids >= self.num_items):
            raise ValueError("history_ids must be in the range [0, num_items)")
        if history_mask is not None:
            if not isinstance(history_mask, torch.Tensor) or history_mask.dtype != torch.bool:
                raise ValueError("history_mask must be a boolean tensor")
            if history_mask.shape != history_ids.shape:
                raise ValueError("history_mask must have the same shape as history_ids")
            if history_mask.device != history_ids.device:
                raise ValueError("history_mask and history_ids must be on the same device")
            if torch.any(history_mask.sum(dim=1) == 0):
                raise ValueError("each history must contain at least one unmasked item")

    def encode(self, history_ids, history_mask=None):
        self._validate_history(history_ids, history_mask)
        item_vectors = self.item_embeddings(history_ids.to(dtype=torch.long))
        if history_mask is None:
            return item_vectors.mean(dim=1)
        weights = history_mask.unsqueeze(-1).to(dtype=item_vectors.dtype)
        return (item_vectors * weights).sum(dim=1) / weights.sum(dim=1)

    def forward(self, history_ids, history_mask=None):
        session_vectors = self.encode(history_ids, history_mask)
        candidates = F.dropout(
            self.item_embeddings.weight, p=self.rho, training=self.training
        )
        normalized_sessions = F.normalize(session_vectors, p=2, dim=-1, eps=1e-8)
        normalized_candidates = F.normalize(candidates, p=2, dim=-1, eps=1e-8)
        return normalized_sessions @ normalized_candidates.transpose(0, 1) / self.tau
