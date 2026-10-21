import torch
from torch import nn


def _invalid_parameter_type(name):
    return ValueError(f"{name} must be an integer")


class TopkMoELayer(nn.Module):
    component_id = "topk-moe-layer"
    PRESETS = {
        "tiny": {"dim": 4, "experts": 2, "topk": 1, "hidden": 4},
        "small": {"dim": 16, "experts": 4, "topk": 2, "hidden": 16},
        "base": {"dim": 64, "experts": 8, "topk": 2, "hidden": 128},
    }

    def __init__(self, dim=32, experts=4, topk=2, hidden=32):
        super().__init__()
        values = {"dim": dim, "experts": experts, "topk": topk, "hidden": hidden}
        for name, value in values.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise _invalid_parameter_type(name)
        if dim <= 0:
            raise ValueError("dim must be positive")
        if experts <= 0:
            raise ValueError("experts must be positive")
        if hidden <= 0:
            raise ValueError("hidden must be positive")
        if not 1 <= topk <= experts:
            raise ValueError("topk must be between 1 and experts")
        self.dim = dim
        self.experts_count = experts
        self.topk = topk
        self.hidden = hidden
        self.router = nn.Linear(dim, experts, bias=False)
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, dim),
            )
            for _ in range(experts)
        )

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim < 1 or x.shape[-1] != self.dim:
            raise ValueError("input must be a tensor whose final dimension equals dim")
        input_shape = x.shape
        flat_x = x.reshape(-1, self.dim)
        logits = self.router(flat_x)
        selected_logits, selected_indices = logits.topk(self.topk, dim=-1)
        selected_gates = selected_logits.softmax(dim=-1)
        output = flat_x.new_zeros((flat_x.shape[0], self.dim))
        for expert_index, expert in enumerate(self.experts):
            assignments = (selected_indices == expert_index).nonzero(as_tuple=False)
            if assignments.shape[0] == 0:
                continue
            rows = assignments[:, 0]
            slots = assignments[:, 1]
            expert_output = expert(flat_x.index_select(0, rows))
            contributions = expert_output * selected_gates[rows, slots].unsqueeze(-1)
            output = output.index_add(0, rows, contributions)
        return output.reshape(input_shape)
