import torch
from torch import nn


class KTO(nn.Module):
    def __init__(self, beta=0.1):
        super().__init__()
        if beta <= 0:
            raise ValueError("beta must be positive")
        self.beta = float(beta)
        self.lambda_desirable = 1.0
        self.lambda_undesirable = 1.0

    def forward(self, policy_logp, reference_logp, desirable, kl_baseline=0.0, reduction="mean"):
        policy_logp = torch.as_tensor(policy_logp)
        reference_logp = torch.as_tensor(reference_logp, device=policy_logp.device, dtype=policy_logp.dtype)
        desirable = torch.as_tensor(desirable, device=policy_logp.device, dtype=torch.bool)
        kl_baseline = torch.as_tensor(kl_baseline, device=policy_logp.device, dtype=policy_logp.dtype)
        if policy_logp.shape != reference_logp.shape:
            raise ValueError("policy_logp and reference_logp must have the same shape")
        if desirable.shape != policy_logp.shape:
            raise ValueError("desirable must have the same shape as log probabilities")
        log_ratio = policy_logp - reference_logp
        baseline = kl_baseline.detach()
        desirable_utility = self.lambda_desirable * torch.sigmoid(self.beta * (log_ratio - baseline))
        undesirable_utility = self.lambda_undesirable * torch.sigmoid(self.beta * (baseline - log_ratio))
        losses = torch.where(
            desirable,
            self.lambda_desirable - desirable_utility,
            self.lambda_undesirable - undesirable_utility,
        )
        if reduction == "none":
            return losses
        if reduction == "sum":
            return losses.sum()
        if reduction == "mean":
            return losses.mean()
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
