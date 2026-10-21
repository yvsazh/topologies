import math

import torch
from torch import nn


class TCVAE(nn.Module):
    architecture_id = "tcvae"
    def __init__(self, latent_dim=4, batch_size=8, dataset_size=None):
        super().__init__()
        if latent_dim < 1 or batch_size < 1:
            raise ValueError("latent_dim and batch_size must be positive")
        if dataset_size is None:
            dataset_size = batch_size
        if dataset_size < batch_size:
            raise ValueError("dataset_size must be at least batch_size")
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.dataset_size = dataset_size

    def decompose(self, mean, log_variance, sample=None):
        if mean.ndim != 2 or log_variance.shape != mean.shape:
            raise ValueError("mean and log_variance must have equal [batch, latent] shapes")
        batch, dimensions = mean.shape
        if dimensions != self.latent_dim or batch < 2:
            raise ValueError("expected at least two samples and configured latent dimensionality")
        z = mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance) if sample is None else sample
        if z.shape != mean.shape:
            raise ValueError("sample must match mean shape")
        log_two_pi = math.log(2.0 * math.pi)
        delta = z.unsqueeze(1) - mean.unsqueeze(0)
        log_density = -0.5 * (log_two_pi + log_variance.unsqueeze(0) + delta.square() * torch.exp(-log_variance.unsqueeze(0)))
        off_diagonal_weight = (self.dataset_size - 1) / (self.dataset_size * (batch - 1))
        weights = torch.full(
            (batch, batch), off_diagonal_weight, device=mean.device, dtype=mean.dtype
        )
        weights.fill_diagonal_(1.0 / self.dataset_size)
        log_weights = weights.log().unsqueeze(-1)
        log_qz = torch.logsumexp(log_density.sum(-1) + log_weights.squeeze(-1), dim=1)
        log_qzi = torch.logsumexp(log_density + log_weights, dim=1)
        log_pz = -0.5 * (log_two_pi + z.square())
        log_q_z_given_x = -0.5 * (log_two_pi + log_variance + (z - mean).square() * torch.exp(-log_variance))
        mutual_information = (log_q_z_given_x.sum(-1) - log_qz).mean()
        total_correlation = (log_qz - log_qzi.sum(-1)).mean()
        dimension_wise_kl = (log_qzi - log_pz).sum(-1).mean()
        return {"mutual_information": mutual_information, "total_correlation": total_correlation,
                "dimension_wise_kl": dimension_wise_kl, "kl": (log_q_z_given_x.sum(-1) - log_pz.sum(-1)).mean(),
                "sample": z}

    def forward(self, mean, log_variance, sample=None):
        return self.decompose(mean, log_variance, sample)
