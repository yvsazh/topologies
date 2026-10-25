import torch
from torch import nn
from torch.nn import functional as F


class Airl(nn.Module):
    component_id = "airl"
    PRESETS = {
        "tiny": {"hidden_features": 8},
        "small": {"hidden_features": 16},
        "base": {"hidden_features": 32},
    }

    def __init__(self, state_features, gamma, hidden_features=32, state_only=True, action_features=None):
        super().__init__()
        if isinstance(state_features, bool) or not isinstance(state_features, int) or state_features <= 0:
            raise ValueError("state_features must be a positive integer")
        if isinstance(hidden_features, bool) or not isinstance(hidden_features, int) or hidden_features <= 0:
            raise ValueError("hidden_features must be a positive integer")
        if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or not 0.0 < gamma < 1.0:
            raise ValueError("gamma must be a number strictly between 0 and 1")
        if type(state_only) is not bool:
            raise ValueError("state_only must be a boolean")
        if action_features is not None and (
            isinstance(action_features, bool) or not isinstance(action_features, int) or action_features <= 0
        ):
            raise ValueError("action_features must be a positive integer when provided")
        if not state_only and action_features is None:
            raise ValueError("action_features is required when state_only is false")
        self.state_features = state_features
        self.gamma = float(gamma)
        self.hidden_features = hidden_features
        self.state_only = state_only
        self.action_features = action_features
        reward_features = state_features if state_only else state_features + action_features
        self.g = nn.Linear(reward_features, 1)
        self.h = nn.Sequential(
            nn.Linear(state_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, 1),
        )

    def _validate_features(self, values, features, name):
        if not torch.is_tensor(values):
            raise ValueError(f"{name} must be a torch.Tensor")
        if values.ndim == 0 or values.shape[-1] != features:
            raise ValueError(f"{name} must end with feature dimension {features}")
        return values.shape[:-1]

    def _reward_input(self, states, actions):
        leading_shape = self._validate_features(states, self.state_features, "states")
        if self.state_only:
            if actions is not None:
                raise ValueError("actions are not accepted when state_only is true")
            return states
        if actions is None:
            raise ValueError("actions are required when state_only is false")
        action_leading_shape = self._validate_features(actions, self.action_features, "actions")
        if action_leading_shape != leading_shape:
            raise ValueError("states and actions must have matching leading dimensions")
        return torch.cat((states, actions), dim=-1)

    def state_reward(self, states, actions=None):
        return self.g(self._reward_input(states, actions)).squeeze(-1)

    def shaped_score(self, states, next_states, actions=None):
        leading_shape = self._validate_features(states, self.state_features, "states")
        next_leading_shape = self._validate_features(next_states, self.state_features, "next_states")
        if next_leading_shape != leading_shape:
            raise ValueError("states and next_states must have matching leading dimensions")
        return self.state_reward(states, actions) + self.gamma * self.h(next_states).squeeze(-1) - self.h(states).squeeze(-1)

    def _normalize_log_policy_prob(self, log_policy_prob, leading_shape):
        if not torch.is_tensor(log_policy_prob):
            raise ValueError("log_policy_prob must be a torch.Tensor")
        if log_policy_prob.shape == leading_shape:
            return log_policy_prob
        if log_policy_prob.ndim == len(leading_shape) + 1 and log_policy_prob.shape[:-1] == leading_shape and log_policy_prob.shape[-1] == 1:
            return log_policy_prob.squeeze(-1)
        if not leading_shape and log_policy_prob.shape == (1,):
            return log_policy_prob.squeeze(-1)
        raise ValueError("log_policy_prob must match transition dimensions with an optional final singleton axis")

    def forward(self, states, next_states, log_policy_prob, actions=None):
        leading_shape = self._validate_features(states, self.state_features, "states")
        log_policy_prob = self._normalize_log_policy_prob(log_policy_prob, leading_shape)
        return self.shaped_score(states, next_states, actions) - log_policy_prob

    def probability(self, states, next_states, log_policy_prob, actions=None):
        return torch.sigmoid(self(states, next_states, log_policy_prob, actions))

    def discriminator_loss(self, expert_logits, policy_logits):
        if not torch.is_tensor(expert_logits) or not torch.is_tensor(policy_logits):
            raise ValueError("expert_logits and policy_logits must be torch.Tensors")
        if expert_logits.numel() == 0 or policy_logits.numel() == 0:
            raise ValueError("expert_logits and policy_logits must be nonempty")
        return F.softplus(-expert_logits).mean() + F.softplus(policy_logits).mean()
