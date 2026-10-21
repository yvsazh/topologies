import torch
from torch import nn
from torch.nn import functional as F


class RWKVBlock(nn.Module):
    def __init__(self, d_model, mode, num_heads):
        super().__init__()
        self.mode = mode
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(d_model)
        self.mix = nn.Parameter(torch.full((d_model,), 0.5))
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.receptance = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        if mode == "rwkv4":
            self.decay = nn.Parameter(torch.zeros(d_model))
            self.bonus = nn.Parameter(torch.zeros(d_model))
        elif mode == "rwkv5":
            self.decay = nn.Parameter(torch.zeros(num_heads))
            self.bonus = nn.Parameter(torch.zeros(num_heads))
        else:
            self.decay = nn.Linear(d_model, d_model)
            self.bonus = nn.Linear(d_model, d_model)
        self.channel_key = nn.Linear(d_model, 4 * d_model)
        self.channel_value = nn.Linear(4 * d_model, d_model)
        self.channel_receptance = nn.Linear(d_model, d_model)

    def _shift(self, x):
        previous = F.pad(x[:, :-1], (0, 0, 1, 0))
        return x * self.mix + previous * (1 - self.mix)

    def _rwkv4(self, x):
        k = self.key(x)
        v = self.value(x)
        r = torch.sigmoid(self.receptance(x))
        decay = torch.exp(-self.decay)
        state = torch.zeros_like(v[:, 0])
        state_num = torch.zeros_like(state)
        state_den = torch.zeros_like(state)
        outputs = []
        for index in range(x.shape[1]):
            bonus = torch.exp(self.bonus + k[:, index])
            numerator = state_num + bonus * v[:, index]
            denominator = state_den + bonus
            outputs.append(r[:, index] * numerator / denominator.clamp_min(1e-8))
            state_num = decay * (state_num + torch.exp(k[:, index]) * v[:, index])
            state_den = decay * (state_den + torch.exp(k[:, index]))
        return self.output(torch.stack(outputs, dim=1))

    def _rwkv5(self, x):
        batch, length, width = x.shape
        head_dim = width // self.num_heads
        k = self.key(x).view(batch, length, self.num_heads, head_dim)
        v = self.value(x).view(batch, length, self.num_heads, head_dim)
        r = torch.sigmoid(self.receptance(x)).view(batch, length, self.num_heads, head_dim)
        state = x.new_zeros(batch, self.num_heads, head_dim, head_dim)
        decay = torch.exp(-self.decay).view(1, self.num_heads, 1, 1)
        outputs = []
        for index in range(length):
            state = state * decay
            state = state + k[:, index].unsqueeze(-1) * v[:, index].unsqueeze(-2)
            read = torch.matmul(r[:, index].unsqueeze(-2), state).squeeze(-2)
            outputs.append(read.reshape(batch, width))
        return self.output(torch.stack(outputs, dim=1))

    def _rwkv6(self, x):
        k = self.key(x)
        v = self.value(x)
        r = torch.sigmoid(self.receptance(x))
        rates = torch.sigmoid(self.decay(x))
        bonus = self.bonus(x)
        state = torch.zeros_like(v[:, 0])
        outputs = []
        for index in range(x.shape[1]):
            rate = rates[:, index]
            state = state * rate + torch.exp(bonus[:, index] + k[:, index]) * v[:, index]
            outputs.append(r[:, index] * state)
        return self.output(torch.stack(outputs, dim=1))

    def forward(self, x):
        residual = x
        normalized = self.norm(x)
        shifted = self._shift(normalized)
        if self.mode == "rwkv4":
            mixed = self._rwkv4(shifted)
        elif self.mode == "rwkv5":
            mixed = self._rwkv5(shifted)
        else:
            mixed = self._rwkv6(shifted)
        x = residual + mixed
        channel = self.norm(x)
        shifted_channel = self._shift(channel)
        key = F.relu(self.channel_key(shifted_channel)).square()
        return x + torch.sigmoid(self.channel_receptance(shifted_channel)) * self.channel_value(key)


class RWKV(nn.Module):
    architecture_id = "rwkv"
    PRESETS = {
        "tiny": {"d_model": 16, "num_layers": 1, "num_heads": 2},
        "base": {"d_model": 64, "num_layers": 4, "num_heads": 4},
    }

    def __init__(self, vocab_size=128, context_length=128, d_model=32, num_layers=2,
                 num_heads=4, mode="rwkv4", dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("RWKV supports dropout=0.0 only")
        if mode not in {"rwkv4", "rwkv5", "rwkv6"}:
            raise ValueError("mode must be rwkv4, rwkv5, or rwkv6")
        if vocab_size <= 1 or context_length <= 0 or d_model <= 0 or num_layers <= 0:
            raise ValueError("model dimensions must be positive and vocab_size must exceed one")
        if num_heads <= 0 or d_model % num_heads:
            raise ValueError("d_model must be divisible by positive num_heads")
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.mode = mode
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([RWKVBlock(d_model, mode, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens):
        if tokens.ndim != 2 or tokens.shape[1] > self.context_length:
            raise ValueError("tokens must have shape (batch, length<=context_length)")
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))
