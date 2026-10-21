import torch
from torch import nn


def leave_one_out_advantages(rewards):
    rewards = torch.as_tensor(rewards)
    if rewards.ndim < 1 or rewards.shape[-1] < 2:
        raise ValueError("each group must contain at least two rewards")
    peer_mean = (rewards.sum(dim=-1, keepdim=True) - rewards) / (rewards.shape[-1] - 1)
    return rewards - peer_mean


class RLOO(nn.Module):
    def __init__(self, group_size=4):
        super().__init__()
        if group_size < 2:
            raise ValueError("group_size must be at least 2")
        self.group_size = int(group_size)

    def advantages(self, rewards):
        rewards = torch.as_tensor(rewards)
        if rewards.shape[-1] != self.group_size:
            raise ValueError("last reward dimension must equal group_size")
        return leave_one_out_advantages(rewards)

    def forward(self, log_probs, rewards):
        log_probs = torch.as_tensor(log_probs)
        rewards = torch.as_tensor(rewards, device=log_probs.device)
        if log_probs.shape != rewards.shape or log_probs.shape[-1] != self.group_size:
            raise ValueError("log_probs and rewards must have matching group-shaped dimensions")
        advantage = leave_one_out_advantages(rewards).to(dtype=log_probs.dtype).detach()
        return -(advantage * log_probs).mean()
