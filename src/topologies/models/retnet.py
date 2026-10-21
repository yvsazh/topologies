import torch
from torch import nn


class RetentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, mode, chunk_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.mode = mode
        self.chunk_size = chunk_size
        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)
        rates = torch.linspace(0.2, 0.8, num_heads)
        self.gamma_logits = nn.Parameter(torch.logit(rates))
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    @property
    def gamma(self):
        return torch.sigmoid(self.gamma_logits)

    def _project(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=-1)
        return tuple(t.view(batch, length, self.num_heads, self.head_dim) for t in (q, k, v))

    def _parallel(self, q, k, v):
        length = q.shape[1]
        i = torch.arange(length, device=q.device).unsqueeze(1)
        j = torch.arange(length, device=q.device).unsqueeze(0)
        distance = i - j
        mask = self.gamma.view(1, self.num_heads, 1, 1).pow(distance.clamp_min(0))
        mask = mask * (distance >= 0).view(1, 1, length, length)
        weights = torch.einsum("bihd,bjhd->bhij", q, k) * mask
        return torch.einsum("bhij,bjhd->bihd", weights, v)

    def _recurrent(self, q, k, v):
        batch, length, _, _ = q.shape
        state = q.new_zeros(batch, self.num_heads, self.head_dim, self.head_dim)
        outputs = []
        gamma = self.gamma.view(1, self.num_heads, 1, 1)
        for index in range(length):
            state = gamma * state + k[:, index].unsqueeze(-1) * v[:, index].unsqueeze(-2)
            outputs.append(torch.matmul(q[:, index].unsqueeze(-2), state).squeeze(-2))
        return torch.stack(outputs, dim=1)

    def _chunkwise(self, q, k, v):
        batch, length, _, _ = q.shape
        state = q.new_zeros(batch, self.num_heads, self.head_dim, self.head_dim)
        outputs = []
        gamma = self.gamma.view(1, self.num_heads, 1, 1)
        for start in range(0, length, self.chunk_size):
            end = min(start + self.chunk_size, length)
            qc, kc, vc = q[:, start:end], k[:, start:end], v[:, start:end]
            count = end - start
            i = torch.arange(count, device=q.device).unsqueeze(1)
            j = torch.arange(count, device=q.device).unsqueeze(0)
            distance = i - j
            mask = self.gamma.view(1, self.num_heads, 1, 1).pow(distance.clamp_min(0))
            mask = mask * (distance >= 0).view(1, 1, count, count)
            local_weights = torch.einsum("bihd,bjhd->bhij", qc, kc) * mask
            local = torch.einsum("bhij,bjhd->bihd", local_weights, vc)
            if start:
                cross = torch.einsum("bihd,bhde->bihe", qc, state)
                local = local + cross * self.gamma.view(1, 1, self.num_heads, 1).pow(torch.arange(1, count + 1, device=q.device).view(1, count, 1, 1))
            outputs.append(local)
            for index in range(count):
                state = gamma * state + kc[:, index].unsqueeze(-1) * vc[:, index].unsqueeze(-2)
        return torch.cat(outputs, dim=1)

    def forward(self, x):
        residual = x
        q, k, v = self._project(x)
        if self.mode == "parallel":
            retained = self._parallel(q, k, v)
        elif self.mode == "recurrent":
            retained = self._recurrent(q, k, v)
        else:
            retained = self._chunkwise(q, k, v)
        x = residual + self.output(retained.reshape_as(x))
        return x + self.ff(self.ff_norm(x))


class RetNet(nn.Module):
    architecture_id = "retnet"
    PRESETS = {"tiny": {"d_model": 16, "num_layers": 1, "num_heads": 2}, "base": {"d_model": 64, "num_layers": 2, "num_heads": 4}}

    def __init__(self, vocab_size=128, context_length=128, d_model=32, num_layers=2, num_heads=4, mode="parallel", chunk_size=2, dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if mode not in {"parallel", "recurrent", "chunkwise"}:
            raise ValueError("mode must be parallel, recurrent, or chunkwise")
        if min(vocab_size, context_length, d_model, num_layers, num_heads, chunk_size) <= 0 or vocab_size <= 1 or d_model % num_heads:
            raise ValueError("invalid RetNet dimensions")
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([RetentionBlock(d_model, num_heads, mode, chunk_size) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens):
        if tokens.ndim != 2 or tokens.shape[1] > self.context_length:
            raise ValueError("tokens must have shape (batch, length<=context_length)")
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))
