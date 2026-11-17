import math

import torch
from torch import nn


class SoftMoE(nn.Module):
    component_id = "soft-moe"

    def __init__(
        self,
        embed_dim=32,
        num_experts=4,
        slots_per_expert=1,
        expert_hidden_dim=None,
        norm_eps=1e-6,
        scale_init=1.0,
    ):
        super().__init__()
        for name, value in (
            ("embed_dim", embed_dim),
            ("num_experts", num_experts),
            ("slots_per_expert", slots_per_expert),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if expert_hidden_dim is not None and (
            isinstance(expert_hidden_dim, bool)
            or not isinstance(expert_hidden_dim, int)
            or expert_hidden_dim <= 0
        ):
            raise ValueError("expert_hidden_dim must be a positive integer or None")
        if isinstance(norm_eps, bool) or not isinstance(norm_eps, (int, float)) or not math.isfinite(norm_eps) or norm_eps <= 0:
            raise ValueError("norm_eps must be a finite positive number")
        if isinstance(scale_init, bool) or not isinstance(scale_init, (int, float)) or not math.isfinite(scale_init):
            raise ValueError("scale_init must be a finite number")
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.norm_eps = float(norm_eps)
        hidden_dim = expert_hidden_dim if expert_hidden_dim is not None else 4 * embed_dim
        self.phi = nn.Parameter(torch.empty(embed_dim, num_experts * slots_per_expert))
        nn.init.xavier_uniform_(self.phi)
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embed_dim),
            )
            for _ in range(num_experts)
        )

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, sequence, embed_dim]")
        if x.shape[0] <= 0 or x.shape[1] <= 0:
            raise ValueError("x batch and sequence dimensions must be positive")
        if x.shape[-1] != self.embed_dim:
            raise ValueError(f"x final dimension must equal embed_dim ({self.embed_dim})")
        if not x.is_floating_point():
            raise ValueError("x must have a floating-point dtype")
        normalized_x = x / (torch.linalg.vector_norm(x, dim=-1, keepdim=True) + self.norm_eps)
        normalized_phi = self.phi / (torch.linalg.vector_norm(self.phi, dim=0, keepdim=True) + self.norm_eps)
        logits = torch.einsum("bld,ds->bls", normalized_x, normalized_phi) * self.scale
        dispatch = torch.softmax(logits, dim=1)
        slot_inputs = torch.einsum("bls,bld->bsd", dispatch, x)
        slot_outputs = torch.cat(
            [
                expert(slot_inputs[:, index * self.slots_per_expert:(index + 1) * self.slots_per_expert])
                for index, expert in enumerate(self.experts)
            ],
            dim=1,
        )
        combine = torch.softmax(logits, dim=-1)
        return torch.einsum("bls,bsd->bld", combine, slot_outputs)
