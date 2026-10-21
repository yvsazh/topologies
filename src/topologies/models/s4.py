import torch
from torch import nn
import torch.nn.functional as F
import math


def hippo_legs(size, dtype=torch.float32, device=None):
    index = torch.arange(size, dtype=dtype, device=device)
    row = index[:, None]
    col = index[None, :]
    return -torch.tril(torch.sqrt(2 * row + 1) * torch.sqrt(2 * col + 1), diagonal=-1) - torch.diag(torch.arange(1, size + 1, dtype=dtype, device=device))


def cauchy_kernel(z, poles):
    return 1 / (z[:, None] - poles[None, :])


class S4Kernel(nn.Module):
    def __init__(self, d_model, state_size, dropout=0.0):
        super().__init__()
        if d_model < 1 or state_size < 1:
            raise ValueError("dimensions must be positive")
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        self.d_model = d_model
        self.state_size = state_size
        legs = hippo_legs(state_size, dtype=torch.float64)
        index = torch.arange(state_size, dtype=torch.float64)
        hippo_p = torch.sqrt(index + 0.5)
        eigenvalues, eigenvectors = torch.linalg.eig(legs + torch.outer(hippo_p, hippo_p))
        transformed_p = eigenvectors.mH @ hippo_p.to(torch.complex128)
        self.register_buffer("eigenvalues", eigenvalues.to(torch.complex64))
        self.register_buffer("rank_left", transformed_p.to(torch.complex64))
        self.register_buffer("rank_right", transformed_p.to(torch.complex64))
        self.B = nn.Parameter(torch.ones(d_model, state_size, dtype=torch.complex64) / state_size ** 0.5)
        self.C = nn.Parameter(torch.ones(d_model, state_size, dtype=torch.complex64) / state_size ** 0.5)
        self.log_dt = nn.Parameter(torch.full((d_model,), -5.0))
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, inputs):
        length = inputs.shape[1]
        dt = self.log_dt.exp().to(self.B.real.dtype)
        poles = 1 + dt[:, None] * self.eigenvalues[None, :]
        left = dt[:, None] * self.rank_left[None, :]
        right = self.rank_right.unsqueeze(0).expand(self.d_model, -1)
        input_factor = dt[:, None] * self.B
        fft_length = max(4096, 16 * length)
        angles = 2 * math.pi * torch.arange(fft_length, dtype=self.B.real.dtype, device=inputs.device) / fft_length
        z = torch.polar(torch.ones_like(angles), angles).to(self.B.dtype)
        responses = []
        for channel in range(self.d_model):
            cauchy = cauchy_kernel(z, poles[channel])
            diagonal_inverse_b = cauchy * input_factor[channel][None, :]
            inverse_left = cauchy * left[channel][None, :]
            denominator = 1 + (inverse_left * right[channel].conj()[None, :]).sum(dim=-1)
            correction = (diagonal_inverse_b * right[channel].conj()[None, :]).sum(dim=-1) / denominator
            responses.append(z * (self.C[channel][None, :] * (diagonal_inverse_b - inverse_left * correction[:, None])).sum(dim=-1))
        kernel = torch.fft.ifft(torch.stack(responses), dim=-1).real[..., :length]
        result = F.conv1d(inputs.transpose(1, 2), kernel.flip(-1).unsqueeze(1), padding=length - 1, groups=self.d_model)[..., :length].transpose(1, 2)
        return result + inputs * self.D


class S4Block(nn.Module):
    def __init__(self, d_model, state_size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.kernel = S4Kernel(d_model, state_size, dropout)
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x):
        return x + torch.sigmoid(self.gate(self.norm(x))) * self.kernel(self.norm(x))


class S4Model(nn.Module):
    architecture_id = "s4"
    PRESETS = {"tiny": {"vocab_size": 64, "d_model": 16, "state_size": 8, "n_layers": 1, "max_seq": 64}, "base": {"vocab_size": 256, "d_model": 128, "state_size": 64, "n_layers": 4, "max_seq": 512}}
    BUDGET_KNOBS = {"d_model": range(16, 257, 16), "n_layers": range(1, 9)}

    def __init__(self, vocab_size=64, d_model=16, state_size=8, n_layers=1, max_seq=64, dropout=0.0):
        super().__init__()
        if min(vocab_size, d_model, state_size, n_layers, max_seq) < 1:
            raise ValueError("model dimensions must be positive")
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        self.vocab_size = vocab_size
        self.max_seq = max_seq
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position = nn.Parameter(torch.zeros(1, max_seq, d_model))
        self.layers = nn.ModuleList([S4Block(d_model, state_size, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, tokens, mask=None):
        if tokens.ndim != 2 or tokens.shape[1] > self.max_seq:
            raise ValueError("tokens must have shape (batch, length) within max_seq")
        x = self.embedding(tokens) + self.position[:, :tokens.shape[1]]
        for layer in self.layers:
            x = layer(x)
        return self.output(self.norm(x))
