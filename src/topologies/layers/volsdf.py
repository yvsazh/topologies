import math

import torch
from torch import nn


class VolSDF(nn.Module):
    component_id = "volsdf"

    def __init__(self, beta_init=0.1):
        super().__init__()
        if isinstance(beta_init, bool) or not isinstance(beta_init, (int, float)):
            self._invalid_argument("beta_init must be a finite positive number.")
        try:
            beta_init = float(beta_init)
        except (OverflowError, ValueError):
            self._invalid_argument("beta_init must be a finite positive number.")
        if not math.isfinite(beta_init) or beta_init <= 0:
            self._invalid_argument("beta_init must be a finite positive number.")
        self.log_beta = nn.Parameter(torch.tensor(math.log(beta_init)))

    @staticmethod
    def _invalid_argument(message):
        raise ValueError(message)

    @property
    def beta(self):
        return self.log_beta.exp()

    def forward(self, sdf):
        if not isinstance(sdf, torch.Tensor):
            self._invalid_argument("sdf must be a floating-point tensor.")
        if not sdf.is_floating_point():
            self._invalid_argument("sdf must be a floating-point tensor.")
        beta = self.log_beta.to(device=sdf.device, dtype=sdf.dtype).exp()
        transformed_sdf = -sdf / beta
        negative_cdf = 0.5 * torch.exp(torch.clamp(transformed_sdf, max=0.0))
        positive_cdf = 1.0 - 0.5 * torch.exp(torch.clamp(-transformed_sdf, max=0.0))
        laplace_cdf = torch.where(transformed_sdf <= 0, negative_cdf, positive_cdf)
        return laplace_cdf / beta
