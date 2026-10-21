import torch
from torch import nn


def ease_weights(gram, l2=1.0):
    gram = torch.as_tensor(gram)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or gram.shape[0] == 0:
        raise ValueError("gram must be a non-empty square matrix")
    if not gram.is_floating_point():
        gram = gram.to(torch.get_default_dtype())
    if l2 <= 0:
        raise ValueError("l2 must be positive")
    regularized = gram + l2 * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    precision = torch.linalg.inv(regularized)
    weights = -precision / torch.diagonal(precision).unsqueeze(0)
    return weights.fill_diagonal_(0)


class EASE(nn.Module):
    architecture_id = "ease"
    def __init__(self, n_items, l2=1.0):
        super().__init__()
        if not isinstance(n_items, int) or n_items <= 0:
            raise ValueError("n_items must be a positive integer")
        if l2 <= 0:
            raise ValueError("l2 must be positive")
        self.n_items = n_items
        self.l2 = float(l2)
        self.register_buffer("weights", torch.zeros(n_items, n_items))
        self.fitted = False

    def fit(self, interactions):
        interactions = torch.as_tensor(interactions)
        if interactions.ndim != 2 or interactions.shape[1] != self.n_items:
            raise ValueError("interactions must have shape (users, n_items)")
        if not interactions.is_floating_point():
            interactions = interactions.to(torch.get_default_dtype())
        gram = interactions.transpose(0, 1) @ interactions
        weights = ease_weights(gram, self.l2)
        self.weights = weights.to(device=self.weights.device, dtype=self.weights.dtype)
        self.fitted = True
        return self

    def forward(self, interactions):
        interactions = torch.as_tensor(interactions, device=self.weights.device)
        if interactions.shape[-1] != self.n_items:
            raise ValueError("last input dimension must equal n_items")
        return interactions.to(dtype=self.weights.dtype) @ self.weights
