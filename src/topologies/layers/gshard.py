import torch
from torch import nn


class _GShardOutput(torch.Tensor):
    def __iter__(self):
        if not hasattr(self, "auxiliary_loss"):
            return super().__iter__()
        return iter((self.as_subclass(torch.Tensor), self.auxiliary_loss))


class GShardMoE(nn.Module):
    component_id = "gshard"

    def __init__(self, embed_dim=4, hidden_dim=8, num_experts=4):
        super().__init__()
        dimensions = (embed_dim, hidden_dim, num_experts)
        if any(type(dimension) is not int for dimension in dimensions):
            raise ValueError("embed_dim, hidden_dim, and num_experts must be integers")
        if embed_dim <= 0 or hidden_dim <= 0 or num_experts < 2:
            raise ValueError("embed_dim and hidden_dim must be positive and num_experts must be at least 2")
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.wg = nn.Parameter(torch.empty(embed_dim, num_experts))
        self.wi = nn.Parameter(torch.empty(num_experts, embed_dim, hidden_dim))
        self.wo = nn.Parameter(torch.empty(num_experts, hidden_dim, embed_dim))
        nn.init.xavier_uniform_(self.wg)
        nn.init.xavier_uniform_(self.wi)
        nn.init.xavier_uniform_(self.wo)

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError("input must be a tensor with shape [groups, group_size, embed_dim]")
        groups, group_size, embed_dim = x.shape
        if groups <= 0 or group_size <= 0 or embed_dim != self.embed_dim:
            raise ValueError("input dimensions must be positive and the feature width must match embed_dim")
        if not x.is_floating_point():
            raise ValueError("input must have a floating-point dtype")
        if 2 * group_size % self.num_experts != 0:
            raise ValueError("2 * group_size / num_experts must be a positive integer")
        capacity = 2 * group_size // self.num_experts
        if capacity <= 0:
            raise ValueError("expert capacity must be positive")

        gates = torch.softmax(torch.einsum("gsm,me->gse", x, self.wg), dim=-1)
        top_gates, top_experts = torch.topk(gates, 2, dim=-1)
        normalized_gates = top_gates / top_gates.sum(dim=-1, keepdim=True)
        dispatch_mask = x.new_zeros((groups, group_size, self.num_experts, capacity))
        combine_weights = gates.new_zeros((groups, group_size, self.num_experts, capacity))
        primary_counts = []

        for group in range(groups):
            counts = [0] * self.num_experts
            for token in range(group_size):
                expert = int(top_experts[group, token, 0])
                slot = counts[expert]
                if slot < capacity:
                    dispatch_mask[group, token, expert, slot] = 1
                    combine_weights[group, token, expert, slot] = normalized_gates[group, token, 0]
                counts[expert] += 1
            primary_counts.append(counts.copy())

            for token in range(group_size):
                expert = int(top_experts[group, token, 1])
                slot = counts[expert]
                draw = torch.rand((), device=x.device)
                if slot < capacity and bool(draw < 2 * normalized_gates[group, token, 1]):
                    dispatch_mask[group, token, expert, slot] = 1
                    combine_weights[group, token, expert, slot] = normalized_gates[group, token, 1]
                counts[expert] += 1

        primary_fractions = x.new_tensor(primary_counts) / group_size
        mean_gates = gates.mean(dim=1)
        auxiliary_loss = (primary_fractions * mean_gates).sum(dim=-1) / self.num_experts
        dispatched_inputs = torch.einsum("gsec,gsm->egcm", dispatch_mask, x)
        hidden = torch.relu(torch.einsum("egcm,emh->egch", dispatched_inputs, self.wi))
        expert_outputs = torch.einsum("egch,ehm->gecm", hidden, self.wo)
        output = torch.einsum("gsec,gecm->gsm", combine_weights, expert_outputs)
        result = output.as_subclass(_GShardOutput)
        result.auxiliary_loss = auxiliary_loss
        return result
