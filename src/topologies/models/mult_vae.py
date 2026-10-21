import torch
from torch import nn
from torch.nn import functional as F


class MultVAE(nn.Module):
    architecture_id = "mult-vae"
    def __init__(self, n_items, latent_dim, hidden_dim, dropout=0.5):
        super().__init__()
        if min(n_items, latent_dim, hidden_dim) <= 0:
            raise ValueError("dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.n_items = n_items
        self.dropout = float(dropout)
        self.encoder = nn.Sequential(nn.Linear(n_items, hidden_dim), nn.Tanh())
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, n_items))

    def encode(self, history):
        normalized = F.normalize(history, p=2, dim=-1)
        hidden = self.encoder(F.dropout(normalized, p=self.dropout, training=self.training))
        return self.mean(hidden), self.logvar(hidden).clamp(-10.0, 10.0)

    @staticmethod
    def kl_divergence(mean, logvar):
        return -0.5 * (1.0 + logvar - mean.square() - logvar.exp()).sum(dim=-1)

    def forward(self, history, sample=None):
        history = torch.as_tensor(history, dtype=self.mean.weight.dtype, device=self.mean.weight.device)
        if history.ndim != 2 or history.shape[-1] != self.n_items:
            raise ValueError("history must have shape (batch, n_items)")
        mean, logvar = self.encode(history)
        if sample is None:
            sample = self.training
        latent = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar) if sample else mean
        logits = self.decoder(latent)
        probabilities = torch.softmax(logits, dim=-1)
        return {"logits": logits, "probabilities": probabilities, "mean": mean, "logvar": logvar, "kl": self.kl_divergence(mean, logvar)}

    def loss(self, history, beta=1.0, sample=None):
        if beta < 0:
            raise ValueError("beta must be nonnegative")
        output = self.forward(history, sample=sample)
        history = torch.as_tensor(history, dtype=output["logits"].dtype, device=output["logits"].device)
        log_likelihood = (history * F.log_softmax(output["logits"], dim=-1)).sum(dim=-1)
        return -(log_likelihood - beta * output["kl"]).mean()
