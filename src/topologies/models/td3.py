import torch
from torch import nn


class TD3(nn.Module):
    architecture_id = "td3"
    def __init__(self, features=4, actions=2, hidden_features=8, policy_delay=2, noise=0.1, noise_clip=0.5):
        super().__init__()
        if features <= 0 or actions <= 0 or hidden_features <= 0:
            raise ValueError("feature and action dimensions must be positive")
        if policy_delay <= 0 or noise < 0 or noise_clip < 0:
            raise ValueError("policy_delay must be positive and noise bounds nonnegative")
        self.features = features
        self.actions = actions
        self.policy_delay = policy_delay
        self.noise = noise
        self.noise_clip = noise_clip
        self.actor = self._network(features, actions, hidden_features, nn.Tanh())
        self.actor_target = self._network(features, actions, hidden_features, nn.Tanh())
        self.critic1 = self._network(features + actions, 1, hidden_features)
        self.critic2 = self._network(features + actions, 1, hidden_features)
        self.critic1_target = self._network(features + actions, 1, hidden_features)
        self.critic2_target = self._network(features + actions, 1, hidden_features)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.register_buffer("update_count", torch.zeros((), dtype=torch.long))
        self.last_target_noise = None
        self.actor_updated = False

    @staticmethod
    def _network(input_features, output_features, hidden_features, output_activation=None):
        layers = [nn.Linear(input_features, hidden_features), nn.ReLU(), nn.Linear(hidden_features, hidden_features), nn.ReLU(), nn.Linear(hidden_features, output_features)]
        if output_activation is not None:
            layers.append(output_activation)
        return nn.Sequential(*layers)

    def forward(self, states):
        return self.actor(states)

    def target_action(self, next_states, noise=None):
        actions = self.actor_target(next_states)
        raw_noise = torch.randn_like(actions) * self.noise if noise is None else torch.as_tensor(noise, dtype=actions.dtype, device=actions.device)
        clipped_noise = raw_noise.clamp(-self.noise_clip, self.noise_clip)
        self.last_target_noise = clipped_noise
        return (actions + clipped_noise).clamp(-1.0, 1.0)

    def target_q(self, next_states, rewards, dones, gamma=0.99, noise=None):
        actions = self.target_action(next_states, noise)
        inputs = torch.cat((next_states, actions), dim=-1)
        values = torch.minimum(self.critic1_target(inputs), self.critic2_target(inputs))
        return rewards + gamma * (1.0 - dones) * values

    def critic_values(self, states, actions):
        inputs = torch.cat((states, actions), dim=-1)
        return self.critic1(inputs), self.critic2(inputs)

    def update_step(self, states, actions, rewards, next_states, dones, critic_optimizer=None, actor_optimizer=None, gamma=0.99):
        self.update_count.add_(1)
        self.actor_updated = False
        target = self.target_q(next_states, rewards, dones, gamma).detach()
        q1, q2 = self.critic_values(states, actions)
        critic_loss = (q1 - target).square().mean() + (q2 - target).square().mean()
        if critic_optimizer is not None:
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
        actor_loss = None
        if int(self.update_count.item()) % self.policy_delay == 0:
            actor_loss = -self.critic1(torch.cat((states, self.actor(states)), dim=-1)).mean()
            if actor_optimizer is not None:
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
            self.actor_updated = True
        return {"critic_loss": critic_loss, "actor_loss": actor_loss, "actor_updated": self.actor_updated}
