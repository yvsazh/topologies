import torch
from torch import nn


class CGAN(nn.Module):
    architecture_id = "cgan"
    def __init__(self, noise_dim=4, num_classes=3, sample_dim=8, embedding_dim=4, hidden_dim=32):
        super().__init__()
        if min(noise_dim, num_classes, sample_dim, embedding_dim, hidden_dim) <= 0:
            raise ValueError("all dimensions must be positive")
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.sample_dim = sample_dim
        self.class_embedding = nn.Embedding(num_classes, embedding_dim)
        self.generator = nn.Sequential(
            nn.Linear(noise_dim + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sample_dim),
            nn.Tanh(),
        )
        self.discriminator = nn.Sequential(
            nn.Linear(sample_dim + embedding_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def _labels(self, labels, batch_size, device):
        labels = torch.as_tensor(labels, dtype=torch.long, device=device)
        if labels.ndim == 0:
            labels = labels.expand(batch_size)
        if labels.ndim != 1 or labels.shape[0] != batch_size:
            raise ValueError("labels must be scalar or one per batch item")
        if torch.any(labels < 0) or torch.any(labels >= self.num_classes):
            raise ValueError("label outside configured class range")
        return labels

    def generate(self, noise, labels):
        if noise.ndim != 2 or noise.shape[1] != self.noise_dim:
            raise ValueError("noise must have shape (batch, noise_dim)")
        labels = self._labels(labels, noise.shape[0], noise.device)
        condition = self.class_embedding(labels)
        return self.generator(torch.cat((noise, condition), dim=1))

    def discriminate(self, samples, labels):
        if samples.ndim != 2 or samples.shape[1] != self.sample_dim:
            raise ValueError("samples must have shape (batch, sample_dim)")
        labels = self._labels(labels, samples.shape[0], samples.device)
        condition = self.class_embedding(labels)
        return self.discriminator(torch.cat((samples, condition), dim=1))

    def forward(self, noise, labels):
        return self.generate(noise, labels)
