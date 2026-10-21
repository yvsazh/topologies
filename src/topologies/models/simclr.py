import torch
from torch import nn
from torch.nn import functional as F


class SimCLR(nn.Module):
    architecture_id = "simclr"
    def __init__(self, input_dim, projection_dim, temperature=0.5):
        super().__init__()
        if input_dim <= 0 or projection_dim <= 0:
            raise ValueError("input_dim and projection_dim must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.encoder = nn.Sequential(nn.Linear(input_dim, projection_dim), nn.ReLU(), nn.Linear(projection_dim, projection_dim))
        self.projector = nn.Sequential(nn.Linear(projection_dim, projection_dim), nn.ReLU(), nn.Linear(projection_dim, projection_dim))
        self.temperature = temperature

    def forward(self, view_one, view_two):
        if view_one.shape != view_two.shape or view_one.ndim != 2:
            raise ValueError("views must have matching (batch, features) shapes")
        if view_one.shape[0] < 2:
            raise ValueError("NT-Xent requires at least two samples")
        if view_one.shape[1] != self.encoder[0].in_features:
            raise ValueError("view feature dimension does not match input_dim")
        batch_size = view_one.shape[0]
        projected = self.projector(self.encoder(torch.cat((view_one, view_two), dim=0)))
        projected = F.normalize(projected, dim=1)
        similarities = projected @ projected.T / self.temperature
        diagonal = torch.eye(2 * batch_size, dtype=torch.bool, device=similarities.device)
        similarities = similarities.masked_fill(diagonal, -torch.inf)
        positives = torch.cat((torch.arange(batch_size, 2 * batch_size), torch.arange(batch_size)), dim=0).to(view_one.device)
        return F.cross_entropy(similarities, positives)
