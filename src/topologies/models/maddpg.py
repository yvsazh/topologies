import torch
from torch import nn


class MADDPG(nn.Module):
    architecture_id = "maddpg"
    def __init__(self, agents=2, features=4, actions=2, hidden_features=8):
        super().__init__()
        if agents <= 0 or features <= 0 or actions <= 0 or hidden_features <= 0:
            raise ValueError("all dimensions must be positive")
        self.agents = agents
        self.features = features
        self.actions = actions
        self.actors = nn.ModuleList([
            nn.Sequential(nn.Linear(features, hidden_features), nn.ReLU(), nn.Linear(hidden_features, actions), nn.Tanh())
            for _ in range(agents)
        ])
        joint = agents * (features + actions)
        self.critics = nn.ModuleList([
            nn.Sequential(nn.Linear(joint, hidden_features), nn.ReLU(), nn.Linear(hidden_features, hidden_features), nn.ReLU(), nn.Linear(hidden_features, 1))
            for _ in range(agents)
        ])

    def actor(self, agent, observation):
        if not 0 <= agent < self.agents:
            raise IndexError("agent index out of range")
        if observation.shape[-1] != self.features:
            raise ValueError("observation last dimension must match features")
        return self.actors[agent](observation)

    def critic(self, agent, observations, actions):
        if not 0 <= agent < self.agents:
            raise IndexError("agent index out of range")
        if observations.shape[-2:] != (self.agents, self.features):
            raise ValueError("observations must end in (agents, features)")
        if actions.shape[-2:] != (self.agents, self.actions):
            raise ValueError("actions must end in (agents, actions)")
        if observations.shape[:-2] != actions.shape[:-2]:
            raise ValueError("observation and action batch dimensions must match")
        joint = torch.cat((observations.flatten(start_dim=-2), actions.flatten(start_dim=-2)), dim=-1)
        return self.critics[agent](joint)

    def forward(self, observations, actions=None):
        if observations.shape[-2:] != (self.agents, self.features):
            raise ValueError("observations must end in (agents, features)")
        actor_actions = torch.stack([self.actor(i, observations[..., i, :]) for i in range(self.agents)], dim=-2)
        if actions is None:
            actions = actor_actions
        values = torch.stack([self.critic(i, observations, actions).squeeze(-1) for i in range(self.agents)], dim=-1)
        return {"actions": actor_actions, "q_values": values}
