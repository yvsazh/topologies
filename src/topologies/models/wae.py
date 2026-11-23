import math

import torch
from torch import nn


class WAE(nn.Module):
    architecture_id = "wae"
    PRESETS = {
        "tiny": {
            "latent_dim": 2,
            "hidden_dim": 16,
            "lambda_mmd": 10.0,
            "prior_variance": 1.0,
            "imq_scales": (1.0,),
        },
        "small": {
            "latent_dim": 8,
            "hidden_dim": 128,
            "lambda_mmd": 10.0,
            "prior_variance": 1.0,
            "imq_scales": (1.0,),
        },
    }

    def __init__(
        self,
        input_dim,
        latent_dim=8,
        hidden_dim=64,
        lambda_mmd=10.0,
        prior_variance=1.0,
        imq_scales=(1.0,),
    ):
        super().__init__()
        for name, value in (
            ("input_dim", input_dim),
            ("latent_dim", latent_dim),
            ("hidden_dim", hidden_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(lambda_mmd, bool)
            or not isinstance(lambda_mmd, (int, float))
            or not math.isfinite(lambda_mmd)
            or lambda_mmd <= 0
        ):
            raise ValueError("lambda_mmd must be a finite positive number")
        if (
            isinstance(prior_variance, bool)
            or not isinstance(prior_variance, (int, float))
            or not math.isfinite(prior_variance)
            or prior_variance <= 0
        ):
            raise ValueError("prior_variance must be a finite positive number")
        try:
            scales = () if isinstance(imq_scales, (str, bytes)) else tuple(imq_scales)
        except TypeError:
            scales = ()
        if not scales or any(
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not math.isfinite(scale)
            or scale <= 0
            for scale in scales
        ):
            raise ValueError("imq_scales must be a non-empty sequence of positive numbers")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lambda_mmd = float(lambda_mmd)
        self.prior_variance = float(prior_variance)
        self.imq_scales = tuple(float(scale) for scale in scales)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, inputs):
        if (
            not isinstance(inputs, torch.Tensor)
            or inputs.ndim != 2
            or inputs.shape[0] == 0
            or inputs.shape[1] != self.input_dim
            or not inputs.is_floating_point()
        ):
            raise ValueError("inputs must be a floating tensor with shape (batch, input_dim)")
        codes = self.encoder(inputs)
        return self.decoder(codes), codes

    def decode(self, codes):
        if (
            not isinstance(codes, torch.Tensor)
            or codes.ndim != 2
            or codes.shape[0] == 0
            or codes.shape[1] != self.latent_dim
            or not codes.is_floating_point()
        ):
            raise ValueError("codes must be a floating tensor with shape (batch, latent_dim)")
        return self.decoder(codes)

    def imq_kernel(self, first, second):
        if (
            not isinstance(first, torch.Tensor)
            or not isinstance(second, torch.Tensor)
            or first.ndim != 2
            or second.ndim != 2
            or first.shape[1] != self.latent_dim
            or second.shape[1] != self.latent_dim
            or not first.is_floating_point()
            or not second.is_floating_point()
        ):
            raise ValueError("kernel samples must have shape (batch, latent_dim)")
        if first.device != second.device or first.dtype != second.dtype:
            raise ValueError("kernel samples must share device and dtype")
        squared_distance = torch.cdist(first, second).square()
        base = 2.0 * self.latent_dim * self.prior_variance
        kernels = [
            scale * base / (scale * base + squared_distance)
            for scale in self.imq_scales
        ]
        return torch.stack(kernels).sum(dim=0)

    def mmd_u_statistic(self, codes, prior_samples):
        self._validate_sample_pair(codes, prior_samples)
        batch_size = codes.shape[0]
        if batch_size < 2:
            raise ValueError("MMD U-statistic requires a batch of at least two samples")
        code_kernel = self.imq_kernel(codes, codes)
        prior_kernel = self.imq_kernel(prior_samples, prior_samples)
        cross_kernel = self.imq_kernel(codes, prior_samples)
        off_diagonal = ~torch.eye(batch_size, dtype=torch.bool, device=codes.device)
        code_term = code_kernel.masked_select(off_diagonal).sum() / (batch_size * (batch_size - 1))
        prior_term = prior_kernel.masked_select(off_diagonal).sum() / (batch_size * (batch_size - 1))
        cross_term = cross_kernel.mean()
        return code_term + prior_term - 2.0 * cross_term

    def objective(self, inputs, reconstruction, codes, prior_samples):
        if (
            not isinstance(inputs, torch.Tensor)
            or inputs.ndim != 2
            or inputs.shape[1] != self.input_dim
            or not isinstance(reconstruction, torch.Tensor)
            or reconstruction.shape != inputs.shape
        ):
            raise ValueError("inputs and reconstruction must share shape (batch, input_dim)")
        if codes.shape[0] != inputs.shape[0]:
            raise ValueError("codes and inputs must have matching batch dimensions")
        reconstruction_loss = (inputs - reconstruction).square().sum(dim=1).mean()
        mmd_loss = self.mmd_u_statistic(codes, prior_samples)
        return reconstruction_loss, mmd_loss, reconstruction_loss + self.lambda_mmd * mmd_loss

    def loss(self, inputs, prior_samples=None):
        reconstruction, codes = self(inputs)
        if prior_samples is None:
            prior_samples = torch.randn_like(codes) * math.sqrt(self.prior_variance)
        else:
            prior_samples = torch.as_tensor(
                prior_samples, device=codes.device, dtype=codes.dtype
            )
        return self.objective(inputs, reconstruction, codes, prior_samples)[2]

    def _validate_sample_pair(self, codes, prior_samples):
        if (
            not isinstance(codes, torch.Tensor)
            or not isinstance(prior_samples, torch.Tensor)
            or codes.ndim != 2
            or prior_samples.ndim != 2
            or codes.shape != prior_samples.shape
            or codes.shape[1] != self.latent_dim
            or not codes.is_floating_point()
            or not prior_samples.is_floating_point()
        ):
            raise ValueError("codes and prior_samples must share shape (batch, latent_dim)")
        if codes.device != prior_samples.device or codes.dtype != prior_samples.dtype:
            raise ValueError("codes and prior_samples must share device and dtype")
