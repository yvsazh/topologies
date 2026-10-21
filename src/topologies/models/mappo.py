import torch
from torch import nn


def clipped_surrogate_loss(new_log_prob, old_log_prob, advantage, epsilon=0.2):
    ratio = torch.exp(new_log_prob - old_log_prob)
    unclipped = ratio * advantage
    clipped = ratio.clamp(1.0 - epsilon, 1.0 + epsilon) * advantage
    return -torch.minimum(unclipped, clipped).mean()


class MAPPO(nn.Module):
    architecture_id = "mappo"
    def __init__(self, agents=2, features=4, num_actions=2, hidden_dim=64, epsilon=0.2):
        super().__init__()
        if agents < 1 or features < 1 or num_actions < 1 or hidden_dim < 1:
            raise ValueError("agents, features, num_actions, and hidden_dim must be positive")
        if not 0.0 < epsilon < 1.0:
            raise ValueError("epsilon must be between zero and one")
        self.agents = agents
        self.features = features
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.actor = nn.Sequential(
            nn.Linear(features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_actions),
        )
        self.value_network = nn.Sequential(
            nn.Linear(agents * features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def policy_logits(self, joint_observation):
        observation = torch.as_tensor(joint_observation, dtype=torch.float32)
        if observation.ndim == 2 and observation.shape == (self.agents, self.features):
            observation = observation.unsqueeze(0)
        if observation.ndim != 3 or observation.shape[1:] != (self.agents, self.features):
            raise ValueError("joint_observation must have shape (batch, agents, features)")
        return self.actor(observation)

    def value(self, joint_observation):
        observation = torch.as_tensor(joint_observation, dtype=torch.float32)
        if observation.ndim == 2 and observation.shape == (self.agents, self.features):
            observation = observation.unsqueeze(0)
        if observation.ndim != 3 or observation.shape[1:] != (self.agents, self.features):
            raise ValueError("joint_observation must have shape (batch, agents, features)")
        return self.value_network(observation.flatten(start_dim=1))

    def forward(self, joint_observation):
        return {
            "policy_logits": self.policy_logits(joint_observation),
            "value": self.value(joint_observation),
        }

    def policy_loss(self, new_log_prob, old_log_prob, advantage):
        return clipped_surrogate_loss(new_log_prob, old_log_prob, advantage, self.epsilon)
