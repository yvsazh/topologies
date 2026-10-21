import torch
import torch.nn as nn
import torch.nn.functional as F


def semiseparable_mask(decay):
    batch, length, heads = decay.shape
    mask = decay.new_zeros(batch, heads, length, length)
    for i in range(length):
        mask[:, :, i, i] = 1
        product = torch.ones(batch, heads, device=decay.device, dtype=decay.dtype)
        for j in range(i - 1, -1, -1):
            product = product * decay[:, j + 1]
            mask[:, :, i, j] = product
    return mask


def mamba2_scan(q, k, v, decay):
    state = torch.zeros_like(v[:, 0])
    outputs = []
    for index in range(v.shape[1]):
        state = decay[:, index, :, None] * state + k[:, index] * v[:, index]
        outputs.append((q[:, index] * state).sum(-1))
    return torch.stack(outputs, dim=1)


def mamba2_dual(q, k, v, decay):
    mask = semiseparable_mask(decay)
    pair_values = torch.einsum("bihd,bjhd,bjhd->bhij", q, k, v)
    return (pair_values * mask).sum(-1).permute(0, 2, 1)


def delta_rule_step(state, key, value, query, alpha, beta):
    identity = torch.eye(key.shape[-1], device=key.device, dtype=key.dtype)
    updated = alpha * state @ (identity - beta * key.unsqueeze(-1) @ key.unsqueeze(-2)) + beta * value.unsqueeze(-1) @ key.unsqueeze(-2)
    return updated, (updated @ query.unsqueeze(-1)).squeeze(-1)


def h3_mixed_output(first_state, multiplicative_gate):
    return first_state * multiplicative_gate


class _Block(nn.Module):
    def __init__(self, width, state_size, mode, heads=4):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.mode = mode
        self.heads = heads
        self.head_dim = width // heads
        self.norm = nn.LayerNorm(width)
        self.in_proj = nn.Linear(width, width * 4)
        self.out_proj = nn.Linear(heads, width) if mode == "mamba2" else nn.Linear(width, width)
        self.decay_proj = nn.Linear(width, 1)
        self.mamba_decay_proj = nn.Linear(width, heads)
        self.state_size = state_size

    def forward(self, x):
        z = self.in_proj(self.norm(x))
        a, b, c, d = z.chunk(4, dim=-1)
        if self.mode == "mamba2":
            shape = (x.shape[0], x.shape[1], self.heads, self.head_dim)
            q = torch.tanh(a).reshape(shape)
            k = torch.tanh(b).reshape(shape)
            v = torch.tanh(c).reshape(shape)
            decay = torch.sigmoid(self.mamba_decay_proj(x))
            self.mamba2_used = (q, k, v, decay)
            y = mamba2_dual(q, k, v, decay)
        elif self.mode == "deltanet":
            state = x.new_zeros(x.shape[0], self.state_size, self.state_size)
            ys = []
            for t in range(x.shape[1]):
                key = F.normalize(a[:, t, :self.state_size], dim=-1)
                query = F.normalize(b[:, t, :self.state_size], dim=-1)
                value = c[:, t, :self.state_size]
                alpha = torch.sigmoid(self.decay_proj(x[:, t]))
                beta = torch.sigmoid(d[:, t, :1])
                state, read = delta_rule_step(state, key, value, query, alpha, beta)
                ys.append(F.pad(read, (0, x.shape[-1] - self.state_size)))
            y = torch.stack(ys, dim=1)
        else:
            first = torch.zeros_like(a[:, 0])
            second = torch.zeros_like(a[:, 0])
            shifted = F.pad(torch.tanh(b[:, :-1]), (0, 0, 1, 0))
            ys = []
            for t in range(x.shape[1]):
                first = torch.sigmoid(self.decay_proj(x[:, t])) * first + shifted[:, t]
                mixed = h3_mixed_output(first, torch.sigmoid(c[:, t]))
                second = torch.sigmoid(d[:, t]) * second + mixed
                ys.append(second)
            y = torch.stack(ys, dim=1)
        return x + self.out_proj(y)


class Mamba2GatedDeltaNetH3(nn.Module):
    architecture_id = "mamba2-gated-deltanet-h3"
    PRESETS = {
        "tiny": dict(vocab_size=128, context_length=128, d_model=32, state_size=16, num_layers=2, mode="mamba2"),
        "base": dict(vocab_size=32128, context_length=2048, d_model=768, state_size=64, num_layers=12, mode="mamba2"),
    }

    def __init__(self, vocab_size=128, context_length=128, d_model=32, state_size=16, num_layers=2, mode="mamba2", dropout=0.0, **kwargs):
        super().__init__()
        if mode not in {"mamba2", "deltanet", "h3"}:
            raise ValueError("mode must be mamba2, deltanet, or h3")
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if min(vocab_size, context_length, d_model, state_size, num_layers) <= 0:
            raise ValueError("model dimensions must be positive")
        self.vocab_size, self.context_length, self.mode = vocab_size, context_length, mode
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([_Block(d_model, state_size, mode) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens):
        if tokens.shape[1] > self.context_length:
            raise ValueError("input exceeds context_length")
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))
