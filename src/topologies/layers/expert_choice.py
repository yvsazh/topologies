import math
from numbers import Integral, Real

import torch
from torch import nn


class ExpertChoice(nn.Module):
    component_id = "expert-choice"
    PRESETS = {
        "tiny": {"width": 8, "num_experts": 2, "capacity_factor": 1.0},
        "small": {"width": 32, "num_experts": 4, "capacity_factor": 2.0},
        "base": {"width": 128, "num_experts": 8, "capacity_factor": 2.0},
    }
    BUDGET_KNOBS = {"width": range(8, 257, 8)}

    def __init__(
        self,
        width: int,
        num_experts: int,
        capacity_factor: float = 2.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(width, bool) or not isinstance(width, Integral) or width <= 0:
            raise ValueError("width must be a positive integer")
        if (
            isinstance(num_experts, bool)
            or not isinstance(num_experts, Integral)
            or num_experts <= 0
        ):
            raise ValueError("num_experts must be a positive integer")
        if (
            isinstance(capacity_factor, bool)
            or not isinstance(capacity_factor, Real)
            or not math.isfinite(float(capacity_factor))
            or capacity_factor <= 0
            or capacity_factor > num_experts
        ):
            raise ValueError("capacity_factor must be finite, positive, and no greater than num_experts")
        if bias is not True and bias is not False:
            raise ValueError("bias must be a boolean")
        self.width = int(width)
        self.num_experts = int(num_experts)
        self.capacity_factor = float(capacity_factor)
        self.gate = nn.Linear(self.width, self.num_experts, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("input must be a torch.Tensor")
        if x.ndim < 2 or x.shape[-1] != self.width:
            raise ValueError("input must have token dimensions and end in width")
        if any(size == 0 for size in x.shape[:-1]):
            raise ValueError("input must contain at least one token")
        if not x.is_floating_point():
            raise ValueError("input must have a floating-point dtype")
        token_shape = x.shape[:-1]
        flat_x = x.reshape(-1, self.width)
        num_tokens = flat_x.shape[0]
        raw_capacity = num_tokens * self.capacity_factor / self.num_experts
        capacity = round(raw_capacity)
        if not math.isclose(raw_capacity, capacity, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("num_tokens * capacity_factor / num_experts must be integral")
        if capacity < 1:
            raise ValueError("expert capacity must be at least one token")
        affinities = self.gate(flat_x).softmax(dim=-1)
        selected_weights, selected_indices = affinities.transpose(0, 1).topk(
            capacity, dim=-1
        )
        dispatch = torch.nn.functional.one_hot(
            selected_indices, num_classes=num_tokens
        ).to(dtype=affinities.dtype)
        routed = torch.einsum("ekn,ek->ne", dispatch, selected_weights)
        return routed.reshape(*token_shape, self.num_experts)
