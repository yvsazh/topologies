import math

import torch
from torch import nn


class CQL(nn.Module):
    architecture_id = "cql"
    PRESETS = {
        "tiny": {"features": 4, "actions": 3, "hidden_features": 8, "conservative_weight": 1.0},
        "small": {"features": 4, "actions": 3, "hidden_features": 64, "conservative_weight": 1.0},
    }

    def __init__(self, features, actions, hidden_features=256, conservative_weight=1.0):
        super().__init__()
        if isinstance(features, bool) or not isinstance(features, int) or features <= 0:
            raise ValueError("features must be a positive integer")
        if isinstance(actions, bool) or not isinstance(actions, int) or actions < 2:
            raise ValueError("actions must be an integer of at least 2")
        if isinstance(hidden_features, bool) or not isinstance(hidden_features, int) or hidden_features <= 0:
            raise ValueError("hidden_features must be a positive integer")
        try:
            if isinstance(conservative_weight, bool) or not isinstance(conservative_weight, (int, float)):
                raise TypeError("conservative_weight must be numeric")
            conservative_weight = float(conservative_weight)
        except (OverflowError, TypeError) as error:
            raise ValueError("conservative_weight must be a finite nonnegative number") from error
        if not math.isfinite(conservative_weight) or conservative_weight < 0:
            raise ValueError("conservative_weight must be a finite nonnegative number")
        self.features = features
        self.actions = actions
        self.hidden_features = hidden_features
        self.conservative_weight = conservative_weight
        self.network = nn.Sequential(
            nn.Linear(features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, actions),
        )

    def forward(self, observation):
        if not isinstance(observation, torch.Tensor):
            raise TypeError("observation must be a torch.Tensor")
        if observation.ndim < 1 or observation.shape[-1] != self.features:
            raise ValueError("observation final dimension must equal features")
        if any(size == 0 for size in observation.shape[:-1]):
            raise ValueError("observation leading dimensions must be nonempty")
        if not observation.is_floating_point():
            raise ValueError("observation must have a floating-point dtype")
        if observation.dtype != self.network[0].weight.dtype:
            raise ValueError("observation dtype must match the model parameter dtype")
        return self.network(observation)

    def loss_components(self, observation, action, target):
        q_values = self(observation)
        try:
            action = torch.as_tensor(action)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError("action must contain integer action indices") from error
        if action.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("action must contain integer action indices")
        if action.shape == q_values.shape[:-1] + (1,):
            action = action.squeeze(-1)
        if action.shape != q_values.shape[:-1]:
            raise ValueError("action shape must match observation leading dimensions")
        action = action.to(device=q_values.device, dtype=torch.long)
        if torch.any(action < 0) or torch.any(action >= self.actions):
            raise ValueError("action index is out of range")
        data_q = q_values.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        try:
            target = torch.as_tensor(target)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError("target must be a floating-point tensor") from error
        if not target.is_floating_point():
            raise ValueError("target must have a floating-point dtype")
        if target.shape != q_values.shape[:-1]:
            raise ValueError("target shape must match observation leading dimensions")
        target = target.to(device=q_values.device, dtype=q_values.dtype).detach()
        td_loss = 0.5 * (data_q - target).square().mean()
        conservative_penalty = (torch.logsumexp(q_values, dim=-1) - data_q).mean()
        loss = td_loss + self.conservative_weight * conservative_penalty
        return {
            "loss": loss,
            "td_loss": td_loss,
            "conservative_penalty": conservative_penalty,
        }

    def loss(self, observation, action, target):
        return self.loss_components(observation, action, target)["loss"]
