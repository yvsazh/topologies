import torch
from torch import nn


class ECAPATDNN(nn.Module):
    architecture_id = "ecapa-tdnn"
    def __init__(self, in_channels=8, num_classes=4, attention_dim=None, eps=1e-5):
        super().__init__()
        if in_channels <= 0 or num_classes <= 0:
            raise ValueError("in_channels and num_classes must be positive")
        attention_dim = attention_dim or max(1, in_channels // 2)
        self.in_channels = in_channels
        self.attention_projection = nn.Linear(in_channels * 3, attention_dim)
        self.attention_score = nn.Linear(attention_dim, in_channels)
        self.classifier = nn.Linear(in_channels * 2, num_classes)
        self.eps = eps
        self.frame_gates = None
        self.embedding = None

    def forward(self, frames):
        if frames.ndim != 3 or frames.shape[-1] != self.in_channels:
            raise ValueError("frames must have shape (batch, time, in_channels)")
        if frames.shape[1] == 0:
            raise ValueError("time dimension must be non-empty")
        context_mean = frames.mean(dim=1)
        context_std = torch.sqrt((frames - context_mean.unsqueeze(1)).square().mean(dim=1) + self.eps)
        context = torch.cat((
            frames,
            context_mean.unsqueeze(1).expand_as(frames),
            context_std.unsqueeze(1).expand_as(frames),
        ), dim=-1)
        scores = self.attention_score(torch.tanh(self.attention_projection(context)))
        gates = torch.softmax(scores, dim=1)
        mean = torch.sum(gates * frames, dim=1)
        second_moment = torch.sum(gates * frames.square(), dim=1)
        variance = (second_moment - mean.square()).clamp_min(self.eps)
        std = torch.sqrt(variance)
        self.frame_gates = gates
        self.embedding = torch.cat((mean, std), dim=-1)
        return self.classifier(self.embedding)
