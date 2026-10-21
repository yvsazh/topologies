import torch
from torch import nn
from torch.nn import functional as F


class DreamerV2(nn.Module):
    architecture_id = "dreamerv2"
    def __init__(self, features=4, actions=3, categories=4, classes=4):
        super().__init__()
        if min(features, actions, categories, classes) <= 0:
            raise ValueError("all dimensions must be positive")
        self.features = features
        self.actions = actions
        self.categories = categories
        self.classes = classes
        self.encoder = nn.Linear(features * 2, categories * classes)
        self.prior = nn.Linear(features + actions, categories * classes)
        self.decoder = nn.Linear(categories * classes, features)

    def posterior_logits(self, observation, state):
        if observation.shape != state.shape or observation.shape[-1] != self.features:
            raise ValueError("observation and state must have matching feature shapes")
        encoded = self.encoder(torch.cat((state, observation), dim=-1))
        return encoded.reshape(*observation.shape[:-1], self.categories, self.classes)

    def prior_logits(self, state, action):
        return self.prior(torch.cat((state, action), dim=-1)).reshape(*state.shape[:-1], self.categories, self.classes)

    def straight_through_sample(self, logits, temperature=1.0, indices=None):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        probabilities = F.softmax(logits / temperature, dim=-1)
        if indices is None:
            indices = torch.distributions.Categorical(probs=probabilities).sample()
        else:
            indices = torch.as_tensor(indices, device=logits.device, dtype=torch.long)
            if indices.shape != logits.shape[:-1] or torch.any(indices < 0) or torch.any(indices >= self.classes):
                raise ValueError("indices must select one valid class per categorical variable")
        hard = F.one_hot(indices, self.classes).to(probabilities.dtype)
        return hard - probabilities.detach() + probabilities

    def kl_divergence(self, posterior_logits, prior_logits):
        posterior = F.softmax(posterior_logits, dim=-1)
        prior = F.softmax(prior_logits, dim=-1)
        return (posterior * (F.log_softmax(posterior_logits, dim=-1) - F.log_softmax(prior_logits, dim=-1))).sum(dim=-1).sum(dim=-1)

    def forward(self, observation, state=None, action=None, temperature=1.0, indices=None):
        if state is None:
            state = torch.zeros_like(observation)
        if action is None:
            action = observation.new_zeros(*observation.shape[:-1], self.actions)
        posterior_logits = self.posterior_logits(observation, state)
        prior_logits = self.prior_logits(state, action)
        latent = self.straight_through_sample(posterior_logits, temperature, indices=indices)
        return {"latent": latent, "reconstruction": self.decoder(latent.flatten(start_dim=-2)), "posterior_logits": posterior_logits, "prior_logits": prior_logits, "kl": self.kl_divergence(posterior_logits, prior_logits)}
