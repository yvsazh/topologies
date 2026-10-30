import torch
from torch import nn


def _invalid_parameter_type(name):
    return ValueError(f"{name} must be an integer")


class BASELayer(nn.Module):
    component_id = "base-layer"
    PRESETS = {
        "tiny": {"num_experts": 2, "embed_dim": 4, "expert_layers": 1},
        "small": {"num_experts": 4, "embed_dim": 16, "expert_layers": 2},
    }

    class _ExpertBlock(nn.Module):
        def __init__(self, embed_dim):
            super().__init__()
            hidden_dim = 4 * embed_dim
            self.norm = nn.LayerNorm(embed_dim)
            self.fc1 = nn.Linear(embed_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, embed_dim)

        def forward(self, x):
            return x + self.fc2(torch.relu(self.fc1(self.norm(x))))

    def __init__(self, num_experts, embed_dim, expert_layers=10):
        super().__init__()
        values = {
            "num_experts": num_experts,
            "embed_dim": embed_dim,
            "expert_layers": expert_layers,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise _invalid_parameter_type(name)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.expert_layers = expert_layers
        self.hidden_dim = 4 * embed_dim
        self.expert_embeddings = nn.Parameter(torch.empty(num_experts, embed_dim))
        nn.init.normal_(self.expert_embeddings, mean=0.0, std=embed_dim**-0.5)
        self.experts = nn.ModuleList(
            nn.ModuleList(self._ExpertBlock(embed_dim) for _ in range(expert_layers))
            for _ in range(num_experts)
        )

    def _balanced_assignment(self, scores):
        if not isinstance(scores, torch.Tensor) or scores.ndim != 2:
            raise ValueError("scores must be a tensor with shape (tokens, experts)")
        token_count, expert_count = scores.shape
        if expert_count != self.num_experts:
            raise ValueError("scores expert dimension must match num_experts")
        if token_count == 0:
            raise ValueError("scores must contain at least one token")
        if token_count % self.num_experts:
            raise ValueError("training token count must be divisible by num_experts")
        if not torch.isfinite(scores).all():
            raise ValueError("scores must contain only finite values")

        capacity = token_count // self.num_experts
        slots = [expert for expert in range(self.num_experts) for _ in range(capacity)]
        values = scores.detach().to(device="cpu", dtype=torch.float64).tolist()
        row_count = token_count
        costs = [[-row[expert] for expert in slots] for row in values]
        u = [0.0] * (row_count + 1)
        v = [0.0] * (row_count + 1)
        matched_row = [0] * (row_count + 1)
        previous_slot = [0] * (row_count + 1)

        for row_index in range(1, row_count + 1):
            matched_row[0] = row_index
            min_cost = [float("inf")] * (row_count + 1)
            used = [False] * (row_count + 1)
            slot_index = 0
            while True:
                used[slot_index] = True
                current_row = matched_row[slot_index]
                delta = float("inf")
                next_slot = 0
                for candidate_slot in range(1, row_count + 1):
                    if used[candidate_slot]:
                        continue
                    reduced_cost = (
                        costs[current_row - 1][candidate_slot - 1]
                        - u[current_row]
                        - v[candidate_slot]
                    )
                    if reduced_cost < min_cost[candidate_slot]:
                        min_cost[candidate_slot] = reduced_cost
                        previous_slot[candidate_slot] = slot_index
                    if min_cost[candidate_slot] < delta:
                        delta = min_cost[candidate_slot]
                        next_slot = candidate_slot
                for candidate_slot in range(row_count + 1):
                    if used[candidate_slot]:
                        u[matched_row[candidate_slot]] += delta
                        v[candidate_slot] -= delta
                    elif candidate_slot > 0:
                        min_cost[candidate_slot] -= delta
                slot_index = next_slot
                if matched_row[slot_index] == 0:
                    break
            while True:
                next_slot = previous_slot[slot_index]
                matched_row[slot_index] = matched_row[next_slot]
                slot_index = next_slot
                if slot_index == 0:
                    break

        assignment = [0] * row_count
        for slot_index in range(1, row_count + 1):
            assignment[matched_row[slot_index] - 1] = slots[slot_index - 1]
        return torch.tensor(assignment, dtype=torch.long, device=scores.device)

    def _assignment_from_scores(self, scores):
        if not torch.isfinite(scores).all():
            raise ValueError("routing scores must contain only finite values")
        if self.training:
            return self._balanced_assignment(scores)
        return scores.argmax(dim=-1)

    def _assign_tokens(self, flat_x):
        if not isinstance(flat_x, torch.Tensor) or flat_x.ndim != 2:
            raise ValueError("flat_x must be a tensor with shape (tokens, embed_dim)")
        if flat_x.shape[-1] != self.embed_dim:
            raise ValueError("flat_x final dimension must match embed_dim")
        scores = flat_x @ self.expert_embeddings.transpose(0, 1)
        return self._assignment_from_scores(scores)

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim < 1:
            raise ValueError("input must be a tensor with a final feature dimension")
        if x.shape[-1] != self.embed_dim:
            raise ValueError("input final dimension must match embed_dim")
        if not x.is_floating_point():
            raise ValueError("input must have a floating point dtype")
        input_shape = x.shape
        flat_x = x.reshape(-1, self.embed_dim)
        if flat_x.shape[0] == 0:
            raise ValueError("input must contain at least one token")
        scores = flat_x @ self.expert_embeddings.transpose(0, 1)
        assignment = self._assignment_from_scores(scores)
        expert_contributions = torch.zeros_like(flat_x)
        for expert_index, blocks in enumerate(self.experts):
            token_indices = (assignment == expert_index).nonzero(as_tuple=True)[0]
            if token_indices.numel() == 0:
                continue
            expert_output = flat_x.index_select(0, token_indices)
            for block in blocks:
                expert_output = block(expert_output)
            gate = torch.sigmoid(scores.index_select(0, token_indices)[:, expert_index]).unsqueeze(-1)
            expert_contributions = expert_contributions.index_add(
                0, token_indices, expert_output * gate
            )
        return (flat_x + expert_contributions).reshape(input_shape)
