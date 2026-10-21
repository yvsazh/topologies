import torch
from torch import nn
from torch.nn import functional as F


class MuZero(nn.Module):
    architecture_id = "muzero"
    def __init__(self, observation_features=4, actions=3, hidden_features=8):
        super().__init__()
        if observation_features <= 0 or actions <= 0 or hidden_features <= 0:
            raise ValueError("feature and action dimensions must be positive")
        self.observation_features = observation_features
        self.actions = actions
        self.hidden_features = hidden_features
        self.representation_network = nn.Sequential(
            nn.Linear(observation_features, hidden_features), nn.ReLU(),
            nn.Linear(hidden_features, hidden_features), nn.Tanh(),
        )
        self.dynamics_network = nn.Sequential(
            nn.Linear(hidden_features + actions, hidden_features), nn.ReLU(),
            nn.Linear(hidden_features, hidden_features + 1),
        )
        self.prediction_network = nn.Sequential(nn.Linear(hidden_features, hidden_features), nn.ReLU())
        self.policy_head = nn.Linear(hidden_features, actions)
        self.value_head = nn.Linear(hidden_features, 1)

    def representation(self, observation):
        observation = torch.as_tensor(observation, dtype=self.representation_network[0].weight.dtype, device=self.representation_network[0].weight.device)
        if observation.shape[-1] != self.observation_features:
            raise ValueError("observation has wrong feature dimension")
        return self.representation_network(observation)

    def dynamics(self, latent, action):
        action = torch.as_tensor(action, device=latent.device)
        if action.ndim == latent.ndim - 1:
            action = F.one_hot(action.long(), self.actions).to(dtype=latent.dtype)
        elif action.shape[-1] == self.actions:
            action = action.to(dtype=latent.dtype)
        else:
            raise ValueError("action must be integer indices or one-hot action vectors")
        prediction = self.dynamics_network(torch.cat((latent, action), dim=-1))
        next_latent, reward = prediction[..., :-1], prediction[..., -1:]
        next_latent = torch.tanh(next_latent)
        return next_latent, reward

    def prediction(self, latent):
        features = self.prediction_network(latent)
        return self.policy_head(features), self.value_head(features).squeeze(-1)

    def forward(self, observation, action=None):
        latent = self.representation(observation)
        policy_logits, value = self.prediction(latent)
        result = {"latent": latent, "policy_logits": policy_logits, "value": value}
        if action is not None:
            next_latent, reward = self.dynamics(latent, action)
            result["next_latent"] = next_latent
            result["reward"] = reward.squeeze(-1)
            result["next_policy_logits"], result["next_value"] = self.prediction(next_latent)
        return result
