import torch
from torch import nn
from torch.nn import functional as F


class WorldModel(nn.Module):
    architecture_id = "world-model"

    def __init__(self, features=4, action_dim=2, deterministic=8, stochastic=4):
        super().__init__()
        if min(features, action_dim, deterministic, stochastic) <= 0:
            raise ValueError("all dimensions must be positive")
        self.features = features
        self.action_dim = action_dim
        self.deterministic_size = deterministic
        self.stochastic_size = stochastic

        self.recurrent = nn.GRUCell(action_dim + stochastic, deterministic)
        self.prior = nn.Linear(deterministic, stochastic * 2)
        self.posterior = nn.Linear(deterministic + features, stochastic * 2)

    @staticmethod
    def _normal_parameters(parameters):
        mean, raw_scale = parameters.chunk(2, dim=-1)
        return mean, F.softplus(raw_scale) + 0.1

    def forward(
        self,
        observation_embedding,
        action,
        deterministic_state=None,
        stochastic_state=None,
        *,
        sample=True,
    ):
        if not torch.is_tensor(observation_embedding):
            observation_embedding = torch.as_tensor(
                observation_embedding, dtype=self.recurrent.weight_ih.dtype
            )
        if not torch.is_tensor(action):
            action = torch.as_tensor(action, dtype=observation_embedding.dtype)
        if observation_embedding.ndim != 2 or observation_embedding.shape[-1] != self.features:
            raise ValueError("observation_embedding must have shape (batch, features)")
        if action.ndim != 2 or action.shape != (observation_embedding.shape[0], self.action_dim):
            raise ValueError("action must have shape (batch, action_dim)")

        batch = observation_embedding.shape[0]
        if deterministic_state is None:
            deterministic_state = observation_embedding.new_zeros(batch, self.deterministic_size)
        if stochastic_state is None:
            stochastic_state = observation_embedding.new_zeros(batch, self.stochastic_size)
        if deterministic_state.shape != (batch, self.deterministic_size):
            raise ValueError("deterministic_state has the wrong shape")
        if stochastic_state.shape != (batch, self.stochastic_size):
            raise ValueError("stochastic_state has the wrong shape")

        hidden = self.recurrent(torch.cat((stochastic_state, action), dim=-1), deterministic_state)
        prior_mean, prior_scale = self._normal_parameters(self.prior(hidden))
        posterior_mean, posterior_scale = self._normal_parameters(
            self.posterior(torch.cat((hidden, observation_embedding), dim=-1))
        )
        if sample:
            latent = posterior_mean + posterior_scale * torch.randn_like(posterior_mean)
        else:
            latent = posterior_mean
        return {
            "deterministic": hidden,
            "stochastic": latent,
            "prior_mean": prior_mean,
            "prior_scale": prior_scale,
            "posterior_mean": posterior_mean,
            "posterior_scale": posterior_scale,
        }
