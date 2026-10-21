from __future__ import annotations

import torch
from torch import nn


class Top2Router(nn.Module):
    component_id = "top-2-router"
    PRESETS = {"tiny": {}, "small": {}, "base": {}}
    BUDGET_KNOBS = {"width": range(8, 257, 8)}

    def __init__(self, width: int, num_experts: int = 8, top_k: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        if width <= 0 or num_experts < 2 or top_k != 2 or top_k > num_experts or not 0 <= dropout < 1:
            raise ValueError("width positive, num_experts >= 2, and top_k=2 required")
        self.num_experts, self.top_k = num_experts, top_k
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(width, num_experts, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        if x.ndim < 2 or x.shape[-1] != self.gate.in_features:
            raise ValueError("input must end in router width")
        shape = x.shape[:-1]
        logits = self.gate(x).float()
        probs = logits.softmax(-1)
        indices = logits.topk(self.top_k, dim=-1).indices
        selected = probs.gather(-1, indices)
        weights = selected / selected.sum(-1, keepdim=True)
        if mask is not None:
            if mask.shape != shape or mask.device != x.device:
                raise ValueError("mask must match input token dimensions and device")
            valid = mask.bool()
            weights = weights * valid.unsqueeze(-1)
            probs = probs * valid.unsqueeze(-1)
            indices = indices.masked_fill(~valid.unsqueeze(-1), -1)
        else:
            valid = torch.ones(shape, dtype=torch.bool, device=x.device)
        flat_probs = probs.reshape(-1, self.num_experts)
        flat_idx, flat_valid = indices.reshape(-1, self.top_k), valid.reshape(-1)
        if bool(flat_valid.any()):
            assignment = torch.nn.functional.one_hot(flat_idx[flat_valid], self.num_experts).float().sum(1).mean(0)
            importance = flat_probs[flat_valid].mean(0)
            aux = self.num_experts * torch.sum(assignment * importance)
        else:
            aux = logits.sum() * 0
        return indices, weights, aux
