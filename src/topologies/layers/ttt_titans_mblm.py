import torch
from torch import nn


class MemoryLayer(nn.Module):
    component_id = "ttt-titans-mblm"
    PRESETS = {
        "tiny": {"d_model": 16, "memory_slots": 4},
        "base": {"d_model": 128, "memory_slots": 32},
    }

    def __init__(self, d_model, memory_slots=8, mode="ttt", learning_rate=0.1, dropout=0.0):
        super().__init__()
        if d_model <= 0 or memory_slots <= 0:
            raise ValueError("d_model and memory_slots must be positive")
        if mode not in {"ttt", "titans", "mblm"}:
            raise ValueError("mode must be ttt, titans, or mblm")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        self.d_model = d_model
        self.memory_slots = memory_slots
        self.mode = mode
        self.learning_rate = learning_rate
        self.dropout = nn.Identity()
        self.memory = nn.Parameter(torch.zeros(d_model, d_model), requires_grad=mode != "mblm")
        self.slots = nn.Parameter(torch.zeros(memory_slots, d_model), requires_grad=mode == "mblm")

    def update(self, weight, key, value):
        prediction = weight @ key
        error = prediction - value
        loss = error.square().sum()
        scale = self.learning_rate
        if self.mode == "titans":
            scale = scale * loss / (1.0 + loss)
        return weight - scale * torch.outer(error, key), loss

    def forward(self, inputs):
        if inputs.ndim != 3 or inputs.shape[-1] != self.d_model:
            raise ValueError("inputs must have shape [batch, length, d_model]")
        if self.mode == "mblm":
            return inputs + self.slots.mean(dim=0).view(1, 1, -1)
        outputs = []
        weight = self.memory
        for token in inputs.unbind(dim=1):
            outputs.append(torch.einsum("ij,bj->bi", weight, token))
            updates = [self.update(weight, key, key)[0] for key in token]
            weight = torch.stack(updates).mean(dim=0)
        return torch.stack(outputs, dim=1)
