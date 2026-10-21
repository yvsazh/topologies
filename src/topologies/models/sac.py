import math

import torch
from torch import nn
from torch.nn import functional as F


class SAC(nn.Module):
    architecture_id = "soft-actor-critic"
    def __init__(self, features=256, actions=6, hidden_features=256, alpha=0.2):
        super().__init__()
        if features <= 0 or actions <= 0 or hidden_features <= 0:
            raise ValueError("features, actions, and hidden_features must be positive")
        if alpha < 0:
            raise ValueError("alpha must be nonnegative")
        self.features = features
        self.actions = actions
        self.alpha = float(alpha)
        self.actor_body = nn.Sequential(nn.Linear(features, hidden_features), nn.ReLU(), nn.Linear(hidden_features, hidden_features), nn.ReLU())
        self.actor_mean = nn.Linear(hidden_features, actions)
        self.actor_log_std = nn.Linear(hidden_features, actions)
        self.critic1 = nn.Sequential(nn.Linear(features + actions, hidden_features), nn.ReLU(), nn.Linear(hidden_features, hidden_features), nn.ReLU(), nn.Linear(hidden_features, 1))
        self.critic2 = nn.Sequential(nn.Linear(features + actions, hidden_features), nn.ReLU(), nn.Linear(hidden_features, hidden_features), nn.ReLU(), nn.Linear(hidden_features, 1))
        self.target_critic1 = nn.Sequential(nn.Linear(features + actions, hidden_features), nn.ReLU(), nn.Linear(hidden_features, hidden_features), nn.ReLU(), nn.Linear(hidden_features, 1))
        self.target_critic2 = nn.Sequential(nn.Linear(features + actions, hidden_features), nn.ReLU(), nn.Linear(hidden_features, hidden_features), nn.ReLU(), nn.Linear(hidden_features, 1))
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        for parameter in self.target_critic1.parameters():
            parameter.requires_grad_(False)
        for parameter in self.target_critic2.parameters():
            parameter.requires_grad_(False)

    def _observation(self, observation):
        observation = torch.as_tensor(observation, dtype=self.actor_mean.weight.dtype, device=self.actor_mean.weight.device)
        if observation.shape[-1] != self.features:
            raise ValueError("last observation dimension must equal features")
        return observation

    def _critic_input(self, observation, action):
        observation = self._observation(observation)
        action = torch.as_tensor(action, dtype=observation.dtype, device=observation.device)
        if action.shape[-1] != self.actions:
            raise ValueError("last action dimension must equal actions")
        return torch.cat((observation, action), dim=-1)

    def sample(self, observation, deterministic=False):
        observation = self._observation(observation)
        hidden = self.actor_body(observation)
        mean = self.actor_mean(hidden)
        log_std = self.actor_log_std(hidden).clamp(-20.0, 2.0)
        if deterministic:
            raw_action = mean
        else:
            raw_action = mean + log_std.exp() * torch.randn_like(mean)
        action = torch.tanh(raw_action)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        log_prob = distribution.log_prob(raw_action) - torch.log(1.0 - action.square() + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)

    def forward(self, observation, action=None):
        observation = self._observation(observation)
        if action is None:
            return self.sample(observation)
        critic_input = self._critic_input(observation, action)
        return self.critic1(critic_input), self.critic2(critic_input)

    def target_value(self, observation, action, log_prob, alpha=None):
        critic_input = self._critic_input(observation, action)
        log_prob = torch.as_tensor(log_prob, dtype=critic_input.dtype, device=critic_input.device)
        temperature = self.alpha if alpha is None else alpha
        q_value = torch.minimum(self.target_critic1(critic_input), self.target_critic2(critic_input))
        return q_value - torch.as_tensor(temperature, dtype=q_value.dtype, device=q_value.device) * log_prob

    @staticmethod
    def soft_bellman_backup(reward, terminal, target_q1, target_q2, log_prob, alpha, gamma):
        target_q1 = torch.as_tensor(target_q1)
        target_q2 = torch.as_tensor(target_q2, dtype=target_q1.dtype, device=target_q1.device)
        log_prob = torch.as_tensor(log_prob, dtype=target_q1.dtype, device=target_q1.device)
        reward = torch.as_tensor(reward, dtype=target_q1.dtype, device=target_q1.device)
        terminal = torch.as_tensor(terminal, dtype=target_q1.dtype, device=target_q1.device)
        soft_value = torch.minimum(target_q1, target_q2) - alpha * log_prob
        return reward + gamma * (1.0 - terminal) * soft_value

    def bellman_backup(
        self, reward, next_observation, next_action, next_log_prob, terminal,
        gamma=0.99, alpha=None,
    ):
        critic_input = self._critic_input(next_observation, next_action)
        q1 = self.target_critic1(critic_input)
        q2 = self.target_critic2(critic_input)
        temperature = self.alpha if alpha is None else alpha
        return self.soft_bellman_backup(
            reward, terminal, q1, q2, next_log_prob, temperature, gamma
        )

    @staticmethod
    def entropy_target(target_q1, target_q2, log_prob, alpha):
        return torch.minimum(target_q1, target_q2) - alpha * log_prob
