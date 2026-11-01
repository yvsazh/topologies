import torch
from torch import nn


class DDPG(nn.Module):
    architecture_id = "ddpg"
    PRESETS = {
        "tiny": {"hidden1": 32, "hidden2": 16},
        "small": {"hidden1": 128, "hidden2": 64},
        "base": {"hidden1": 400, "hidden2": 300},
    }

    def __init__(self, observation_dim, action_dim, hidden1=400, hidden2=300):
        super().__init__()
        dimensions = {
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "hidden1": hidden1,
            "hidden2": hidden2,
        }
        for name, value in dimensions.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2

        self.actor_state_norm = nn.BatchNorm1d(observation_dim)
        self.actor_fc1 = nn.Linear(observation_dim, hidden1)
        self.actor_bn1 = nn.BatchNorm1d(hidden1)
        self.actor_fc2 = nn.Linear(hidden1, hidden2)
        self.actor_bn2 = nn.BatchNorm1d(hidden2)
        self.actor_output = nn.Linear(hidden2, action_dim)

        self.critic_state_norm = nn.BatchNorm1d(observation_dim)
        self.critic_fc1 = nn.Linear(observation_dim, hidden1)
        self.critic_bn1 = nn.BatchNorm1d(hidden1)
        self.critic_fc2 = nn.Linear(hidden1 + action_dim, hidden2)
        self.critic_output = nn.Linear(hidden2, 1)

        for layer in (
            self.actor_fc1,
            self.actor_fc2,
            self.critic_fc1,
            self.critic_fc2,
        ):
            bound = 1.0 / layer.in_features**0.5
            nn.init.uniform_(layer.weight, -bound, bound)
            nn.init.uniform_(layer.bias, -bound, bound)
        for layer in (self.actor_output, self.critic_output):
            nn.init.uniform_(layer.weight, -3e-3, 3e-3)
            nn.init.uniform_(layer.bias, -3e-3, 3e-3)

    def _validate_observations(self, observations):
        if not isinstance(observations, torch.Tensor):
            raise TypeError("observations must be a torch.Tensor")
        if observations.ndim != 2 or observations.shape[1] != self.observation_dim:
            raise ValueError(
                f"observations must have shape [B, {self.observation_dim}]"
            )
        if not observations.is_floating_point():
            raise ValueError("observations must have a floating-point dtype")

    def actor(self, observations):
        self._validate_observations(observations)
        hidden = self.actor_state_norm(observations)
        hidden = torch.relu(self.actor_bn1(self.actor_fc1(hidden)))
        hidden = torch.relu(self.actor_bn2(self.actor_fc2(hidden)))
        return torch.tanh(self.actor_output(hidden))

    def critic(self, observations, actions):
        self._validate_observations(observations)
        if not isinstance(actions, torch.Tensor):
            raise TypeError("actions must be a torch.Tensor")
        if actions.ndim != 2 or actions.shape[1] != self.action_dim:
            raise ValueError(f"actions must have shape [B, {self.action_dim}]")
        if actions.shape[0] != observations.shape[0]:
            raise ValueError("observations and actions must have the same batch size")
        if not actions.is_floating_point():
            raise ValueError("actions must have a floating-point dtype")
        hidden = self.critic_state_norm(observations)
        hidden = torch.relu(self.critic_bn1(self.critic_fc1(hidden)))
        hidden = torch.cat((hidden, actions), dim=1)
        hidden = torch.relu(self.critic_fc2(hidden))
        return self.critic_output(hidden)

    def forward(self, observations, actions=None):
        predicted_actions = self.actor(observations)
        critic_actions = predicted_actions if actions is None else actions
        q_values = self.critic(observations, critic_actions)
        return {"actions": predicted_actions, "q_values": q_values}
