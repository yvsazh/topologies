import torch
from torch import nn
from torch.distributions import Normal


def e_step_weights(q_values, temperature, prior_log_probabilities=None):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = q_values / temperature
    if prior_log_probabilities is not None:
        if prior_log_probabilities.shape != q_values.shape:
            raise ValueError("prior_log_probabilities must match q_values")
        logits = logits + prior_log_probabilities
    return torch.softmax(logits, dim=-1)


def m_step_loss(log_probabilities, weights):
    if weights.shape != log_probabilities.shape:
        raise ValueError("weights must match log_probabilities")
    return -(weights.detach() * log_probabilities).sum(dim=-1).mean()


class MPO(nn.Module):
    architecture_id = "mpo"
    def __init__(self, features=4, actions=3, hidden_features=8, temperature=1.0):
        super().__init__()
        if features <= 0 or actions <= 0 or hidden_features <= 0 or temperature <= 0:
            raise ValueError("dimensions and temperature must be positive")
        self.actions = actions
        self.temperature = temperature
        self.encoder = nn.Sequential(nn.Linear(features, hidden_features), nn.ReLU())
        self.mean = nn.Linear(hidden_features, actions)
        self.log_std = nn.Linear(hidden_features, actions)

    def forward(self, observations):
        hidden = self.encoder(observations)
        return Normal(self.mean(hidden), self.log_std(hidden).clamp(-20.0, 2.0).exp())

    def e_step(self, q_values, prior_log_probabilities=None):
        return e_step_weights(q_values, self.temperature, prior_log_probabilities)

    def m_step_loss(self, observations, sampled_actions, weights):
        policy = self(observations)
        if sampled_actions.ndim == observations.ndim + 1:
            policy = Normal(policy.loc.unsqueeze(-2), policy.scale.unsqueeze(-2))
        log_probabilities = policy.log_prob(sampled_actions).sum(dim=-1)
        return m_step_loss(log_probabilities, weights)
