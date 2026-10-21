import torch
from torch import nn
from torch.nn import functional as F


class ICM(nn.Module):
    architecture_id = "icm"
    def __init__(self, feature_dim=4, action_dim=3, hidden_features=8):
        super().__init__()
        if feature_dim <= 0 or action_dim <= 0 or hidden_features <= 0:
            raise ValueError("feature_dim, action_dim, and hidden_features must be positive")
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, action_dim),
        )
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, feature_dim),
        )

    def forward(self, state, next_state, action):
        if state.shape[-1] != self.feature_dim or next_state.shape != state.shape:
            raise ValueError("state and next_state must have matching feature dimensions")
        if action.shape[-1] != self.action_dim:
            raise ValueError("action must have the configured action dimension")
        predicted_action = self.inverse_model(torch.cat((state, next_state), dim=-1))
        predicted_next = self.forward_model(torch.cat((state, action), dim=-1))
        bonus = self.curiosity_bonus(predicted_next, next_state)
        return {"predicted_action": predicted_action, "predicted_next_state": predicted_next, "bonus": bonus}

    @staticmethod
    def curiosity_bonus(predicted_next_state, next_state):
        return 0.5 * (predicted_next_state - next_state).square().sum(dim=-1)

    def inverse_loss(self, state, next_state, action):
        predicted_action = self.inverse_model(torch.cat((state, next_state), dim=-1))
        return F.cross_entropy(predicted_action.reshape(-1, self.action_dim), action.reshape(-1).long())

    def forward_loss(self, state, next_state, action):
        predicted_next = self.forward_model(torch.cat((state, action), dim=-1))
        return 0.5 * (predicted_next - next_state).square().sum(dim=-1).mean()
