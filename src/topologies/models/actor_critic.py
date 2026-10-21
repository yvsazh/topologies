import torch
from torch import nn

class ActorCritic(nn.Module):
    architecture_id = 'actor-critic'

    def __init__(self, observation_dim, action_count, hidden_dims=(64, 64), shared_trunk=True):
        super().__init__()
        if observation_dim < 1 or action_count < 2 or (not hidden_dims) or any((w < 1 for w in hidden_dims)):
            raise ValueError('positive observation and hidden sizes and at least two actions are required')

        def mlp():
            modules, width = ([], observation_dim)
            for size in hidden_dims:
                modules.extend((nn.Linear(width, size), nn.Tanh()))
                width = size
            return (nn.Sequential(*modules), width)
        self.shared_trunk = shared_trunk
        if shared_trunk:
            self.trunk, width = mlp()
            self.actor = nn.Linear(width, action_count)
            self.critic = nn.Linear(width, 1)
        else:
            self.actor_trunk, width = mlp()
            self.critic_trunk, critic_width = mlp()
            self.actor = nn.Linear(width, action_count)
            self.critic = nn.Linear(critic_width, 1)

    def forward(self, observations):
        if self.shared_trunk:
            features = self.trunk(observations)
            return (self.actor(features), self.critic(features).squeeze(-1))
        return (self.actor(self.actor_trunk(observations)), self.critic(self.critic_trunk(observations)).squeeze(-1))
