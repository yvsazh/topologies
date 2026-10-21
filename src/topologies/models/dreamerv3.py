import torch
from torch import nn
from torch.nn import functional as F


class DreamerV3(nn.Module):
    architecture_id = "dreamerv3"
    def __init__(self, features, bins, hidden_features=128, symlog_limit=20.0):
        super().__init__()
        if features < 1 or bins < 2 or hidden_features < 1:
            raise ValueError("features and hidden_features must be positive; bins must be at least two")
        if symlog_limit <= 0:
            raise ValueError("symlog_limit must be positive")
        self.features = features
        self.bins = bins
        self.symlog_limit = float(symlog_limit)
        self.reward_head = nn.Sequential(nn.Linear(features, hidden_features), nn.ELU(), nn.Linear(hidden_features, bins))
        self.register_buffer("support", torch.linspace(-self.symlog_limit, self.symlog_limit, bins))

    @staticmethod
    def symlog(value):
        return torch.sign(value) * torch.log1p(torch.abs(value))

    @staticmethod
    def symexp(value):
        return torch.sign(value) * torch.expm1(torch.abs(value))

    def two_hot(self, reward):
        transformed = self.symlog(reward).clamp(-self.symlog_limit, self.symlog_limit)
        position = (transformed + self.symlog_limit) * (self.bins - 1) / (2 * self.symlog_limit)
        lower = position.floor().long().clamp(0, self.bins - 1)
        upper = (lower + 1).clamp(max=self.bins - 1)
        upper_weight = position - lower.to(position.dtype)
        upper_weight = torch.where(lower == upper, torch.zeros_like(upper_weight), upper_weight)
        target = torch.zeros(*reward.shape, self.bins, dtype=reward.dtype, device=reward.device)
        target.scatter_add_(-1, lower.unsqueeze(-1), (1 - upper_weight).unsqueeze(-1))
        target.scatter_add_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
        return target

    def reward_loss(self, features, reward):
        logits = self.reward_head(features)
        if logits.shape[:-1] != reward.shape:
            raise ValueError("reward shape must match features leading dimensions")
        target = self.two_hot(reward)
        return -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

    def predict_reward(self, features):
        probabilities = self.reward_head(features).softmax(dim=-1)
        return self.symexp((probabilities * self.support).sum(dim=-1))

    def forward(self, features):
        return self.reward_head(features)
