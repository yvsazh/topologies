import torch
from torch import nn


class MemorizingAttention(nn.Module):
    component_id = "memorizing-attention"

    def __init__(self, d_model=32, neighbors=1, segment_size=2, mode="memorizing", dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if mode not in {"memorizing", "infini"}:
            raise ValueError("mode must be memorizing or infini")
        if min(d_model, neighbors, segment_size) <= 0:
            raise ValueError("invalid memory dimensions")
        self.d_model = d_model
        self.neighbors = neighbors
        self.segment_size = segment_size
        self.mode = mode
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, 1, bias=False)
        if mode == "memorizing":
            self.query.requires_grad_(False)
            self.key.requires_grad_(False)
            self.gate.requires_grad_(False)

    def compressive_update(self, key, value, memory, normalizer):
        positive = torch.nn.functional.elu(key) + 1
        return memory + torch.outer(positive, value), normalizer + positive

    def compressive_retrieve(self, query, memory, normalizer):
        positive = torch.nn.functional.elu(query) + 1
        numerator = positive @ memory
        denominator = (positive * normalizer).sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return numerator / denominator

    def _memorizing(self, hidden):
        query = self.query(hidden)
        key = self.key(hidden)
        value = self.value(hidden)
        output = value.clone()
        steps = hidden.shape[1]
        for index in range(1, steps):
            past_key = key[:, :index]
            past_value = value[:, :index]
            distance = (query[:, index, None, :] - past_key).pow(2).sum(dim=-1)
            count = min(self.neighbors, index)
            chosen = distance.topk(count, dim=-1, largest=False).indices
            gathered = past_value.gather(1, chosen[..., None].expand(-1, -1, self.d_model))
            output[:, index] = gathered.mean(dim=1)
        return output

    def _infini(self, hidden):
        query = self.query(hidden)
        key = self.key(hidden)
        value = self.value(hidden)
        batch, steps, width = hidden.shape
        memory = hidden.new_zeros(batch, width, width)
        normalizer = hidden.new_zeros(batch, width)
        output = value.clone()
        for start in range(0, steps, self.segment_size):
            stop = min(steps, start + self.segment_size)
            retrieved = []
            for row in range(batch):
                retrieved.append(self.compressive_retrieve(query[row, start:stop], memory[row], normalizer[row]))
            retrieved = torch.stack(retrieved)
            mix = torch.sigmoid(self.gate(hidden[:, start:stop]))
            output[:, start:stop] = mix * retrieved + (1.0 - mix) * value[:, start:stop]
            for row in range(batch):
                updated_memory, updated_normalizer = memory[row], normalizer[row]
                for offset in range(stop - start):
                    updated_memory, updated_normalizer = self.compressive_update(
                        key[row, start + offset], value[row, start + offset], updated_memory, updated_normalizer)
                memory[row] = updated_memory
                normalizer[row] = updated_normalizer
        return output

    def forward(self, hidden):
        hidden = torch.as_tensor(hidden)
        if hidden.ndim != 3 or hidden.shape[-1] != self.d_model:
            raise ValueError("hidden must have shape (batch, length, d_model)")
        if self.mode == "memorizing":
            return self._memorizing(hidden)
        return self._infini(hidden)
