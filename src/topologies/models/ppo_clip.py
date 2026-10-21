import torch
from torch import nn


class PPOClip(nn.Module):
    def __init__(self, epsilon=0.2):
        super().__init__()
        if not 0.0 < epsilon < 1.0:
            raise ValueError("epsilon must be between zero and one")
        self.epsilon = float(epsilon)

    def forward(self, new_log_prob, old_log_prob, advantage):
        new_log_prob = torch.as_tensor(new_log_prob)
        old_log_prob = torch.as_tensor(old_log_prob, device=new_log_prob.device, dtype=new_log_prob.dtype)
        advantage = torch.as_tensor(advantage, device=new_log_prob.device, dtype=new_log_prob.dtype)
        ratio = torch.exp(new_log_prob - old_log_prob)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
        return torch.minimum(ratio * advantage, clipped_ratio * advantage)
