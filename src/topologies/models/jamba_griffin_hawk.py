import math

import torch
from torch import nn
from torch.nn import functional as F


class RealGatedRecurrence(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gates = nn.Linear(d_model, 2 * d_model)

    def gate_values(self, x):
        pre_a, pre_i = self.gates(x).chunk(2, dim=-1)
        return torch.sigmoid(pre_a), torch.sigmoid(pre_i)

    def forward(self, x):
        batch, length, width = x.shape
        state = x.new_zeros(batch, width)
        outputs = []
        for index in range(length):
            a, i = self.gate_values(x[:, index])
            state = a * state + torch.sqrt((1.0 - a.square()).clamp_min(0.0)) * (i * x[:, index])
            outputs.append(state)
        return torch.stack(outputs, dim=1)


class SelectiveRecurrence(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gates = nn.Linear(d_model, 3 * d_model)

    def forward(self, x):
        batch, length, width = x.shape
        state = x.new_zeros(batch, width)
        outputs = []
        for index in range(length):
            delta, decay, injection = self.gates(x[:, index]).chunk(3, dim=-1)
            step = F.softplus(delta)
            a = torch.exp(-F.softplus(decay) * step)
            state = a * state + (1.0 - a) * torch.tanh(injection)
            outputs.append(state)
        return torch.stack(outputs, dim=1)


class CausalAttention(nn.Module):
    def __init__(self, d_model, num_heads, window=None):
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.window = window
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, length, width = x.shape
        head_width = width // self.num_heads
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, self.num_heads, head_width).transpose(1, 2)
        k = k.view(batch, length, self.num_heads, head_width).transpose(1, 2)
        v = v.view(batch, length, self.num_heads, head_width).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_width)
        positions = torch.arange(length, device=x.device)
        allowed = positions.unsqueeze(0) <= positions.unsqueeze(1)
        if self.window is not None:
            allowed = allowed & ((positions.unsqueeze(1) - positions.unsqueeze(0)) < self.window)
        weights = torch.softmax(scores.masked_fill(~allowed, torch.finfo(scores.dtype).min), dim=-1)
        values = torch.matmul(weights, v).transpose(1, 2).reshape(batch, length, width)
        return self.output(values)


class RoutedExperts(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.router = nn.Linear(d_model, 2)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
            for _ in range(2)
        ])

    def forward(self, x):
        weights = torch.softmax(self.router(x), dim=-1)
        values = torch.stack([expert(x) for expert in self.experts], dim=-2)
        return (weights.unsqueeze(-1) * values).sum(dim=-2)


class HybridLayer(nn.Module):
    def __init__(self, d_model, kind, num_heads, window, routed):
        super().__init__()
        self.kind = kind
        self.norm = nn.LayerNorm(d_model)
        if kind == "attention":
            self.mixer = CausalAttention(d_model, num_heads)
        elif kind == "local_attention":
            self.mixer = CausalAttention(d_model, num_heads, window)
        elif kind == "selective_recurrence":
            self.mixer = SelectiveRecurrence(d_model)
        else:
            self.mixer = RealGatedRecurrence(d_model)
        self.mixer_output = nn.Linear(d_model, d_model)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = RoutedExperts(d_model) if routed else nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        x = x + self.mixer_output(self.mixer(self.norm(x)))
        return x + self.ff(self.ff_norm(x))


class JambaGriffinHawk(nn.Module):
    architecture_id = "jamba-griffin-hawk"
    PRESETS = {
        "tiny": {"d_model": 16, "num_layers": 2, "num_heads": 2},
        "base": {"d_model": 64, "num_layers": 4, "num_heads": 4},
    }

    def __init__(self, vocab_size=128, context_length=128, d_model=32, num_layers=4,
                 num_heads=4, mode="jamba", window=4, dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if mode not in {"jamba", "griffin", "hawk"}:
            raise ValueError("mode must be jamba, griffin, or hawk")
        if min(vocab_size, context_length, d_model, num_layers, num_heads, window) <= 0 or vocab_size <= 1:
            raise ValueError("invalid model dimensions")
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.mode = mode
        if mode == "jamba":
            schedule = ["selective_recurrence" if index % 2 == 0 else "attention" for index in range(num_layers)]
        elif mode == "griffin":
            schedule = ["real_gated_recurrence" if index % 2 == 0 else "local_attention" for index in range(num_layers)]
        else:
            schedule = ["real_gated_recurrence"] * num_layers
        self.schedule = tuple(schedule)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            HybridLayer(d_model, kind, num_heads, window, mode == "jamba" and index % 2 == 1)
            for index, kind in enumerate(schedule)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens):
        if tokens.ndim != 2 or tokens.shape[1] > self.context_length:
            raise ValueError("tokens must have shape (batch, length<=context_length)")
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))
