import torch
from torch import nn
import torch.nn.functional as F

def mu_law_encode(waveform, quantization_channels=256):
    mu = quantization_channels - 1
    companded = torch.sign(waveform) * torch.log1p(mu * waveform.abs().clamp(max=1)) / torch.log(torch.tensor(1.0 + mu, device=waveform.device))
    return ((companded + 1) * 0.5 * mu + 0.5).long().clamp(0, mu)

def mu_law_decode(values, quantization_channels=256):
    mu = quantization_channels - 1
    y = 2.0 * values.to(torch.float32) / mu - 1.0
    return torch.sign(y) * torch.expm1(y.abs() * torch.log(torch.tensor(1.0 + mu, device=y.device))) / mu

def shifted_targets(tokens):
    if tokens.shape[-1] < 2:
        raise ValueError('sequence needs at least two tokens')
    return (tokens[..., :-1], tokens[..., 1:])

class _CausalBlock(nn.Module):

    def __init__(self, width, dilation, cond_dim=0):
        super().__init__()
        self.dilation = dilation
        self.conv = nn.Conv1d(width, width * 2, 2, dilation=dilation)
        self.cond = nn.Linear(cond_dim, width * 2) if cond_dim else None
        self.res = nn.Conv1d(width, width, 1)
        self.skip = nn.Conv1d(width, width, 1)

    def forward(self, x, condition=None):
        z = self.conv(F.pad(x, (self.dilation, 0)))
        if self.cond is not None:
            if condition is None:
                raise ValueError('conditioning required')
            z = z + self.cond(condition).transpose(1, 2)
        a, b = z.chunk(2, 1)
        z = torch.tanh(a) * torch.sigmoid(b)
        return (x + self.res(z), self.skip(z))

class WaveNet(nn.Module):
    architecture_id = 'wavenet'

    @classmethod
    def preset(cls, name, **geometry):
        widths = {'tiny': (16, 4), 'small': (32, 8), 'base': (64, 12)}
        if name not in widths:
            raise ValueError('preset must be tiny, small, or base')
        channels, layers = widths[name]
        return cls(residual_channels=channels, layers=layers, **geometry)

    def __init__(self, quantization_channels=256, residual_channels=32, layers=8, stacks=1, conditioning_dim=0):
        super().__init__()
        if min(quantization_channels, residual_channels, layers, stacks) <= 0 or conditioning_dim < 0:
            raise ValueError('invalid dimensions')
        self.quantization_channels = quantization_channels
        self.embedding = nn.Embedding(quantization_channels, residual_channels)
        self.blocks = nn.ModuleList([_CausalBlock(residual_channels, 2 ** (i % (layers // stacks or 1)), conditioning_dim) for i in range(layers)])
        self.proj = nn.Sequential(nn.ReLU(), nn.Conv1d(residual_channels, residual_channels, 1), nn.ReLU(), nn.Conv1d(residual_channels, quantization_channels, 1))

    def forward(self, tokens, conditioning=None):
        if tokens.ndim != 2:
            raise ValueError('tokens must be [B,T]')
        if torch.any(tokens < 0) or torch.any(tokens >= self.quantization_channels):
            raise ValueError('token outside quantizer')
        x = self.embedding(tokens).transpose(1, 2)
        skips = []
        for block in self.blocks:
            x, s = block(x, conditioning)
            skips.append(s)
        return self.proj(torch.stack(skips).sum(0)).transpose(1, 2)

    @torch.no_grad()
    def generate(self, prefix, num_samples, conditioning=None, temperature=1.0):
        if num_samples < 0 or temperature <= 0:
            raise ValueError('invalid generation arguments')
        tokens = prefix
        for i in range(num_samples):
            logits = self(tokens, conditioning)[:, -1] / temperature
            tokens = torch.cat([tokens, torch.multinomial(logits.softmax(-1), 1)], 1)
        return tokens
