import math

import torch
from torch import nn


class VESDEPrimitives(nn.Module):
    component_id = "score-sde"

    def __init__(self, sigma=0.5, sigma_min=0.01, sigma_max=50.0):
        super().__init__()
        if sigma <= 0 or sigma_min <= 0 or sigma_max <= sigma_min:
            raise ValueError("noise scales must be positive and ordered")
        self.sigma = float(sigma)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

    def noise_level(self, time):
        time = torch.as_tensor(time)
        return self.sigma_min * torch.pow(self.sigma_max / self.sigma_min, time)

    def diffusion_coefficient(self, time):
        return self.noise_level(time) * math.sqrt(2.0 * math.log(self.sigma_max / self.sigma_min))

    def perturb(self, sample, noise, sigma=None):
        if sample.shape != noise.shape:
            raise ValueError("sample and noise must have the same shape")
        scale = self.sigma if sigma is None else sigma
        return sample + torch.as_tensor(scale, dtype=sample.dtype, device=sample.device) * noise

    def probability_flow_step(self, sample, score, step_size, sigma=None, time=None):
        if sigma is not None and time is not None:
            raise ValueError("provide either a diffusion coefficient sigma or time, not both")
        if time is not None:
            scale = self.diffusion_coefficient(
                torch.as_tensor(time, dtype=sample.dtype, device=sample.device)
            )
        else:
            scale = self.sigma if sigma is None else sigma
            scale = torch.as_tensor(scale, dtype=sample.dtype, device=sample.device)
        return sample - 0.5 * scale.square() * step_size * score

    def forward(self, sample, noise=None, sigma=None):
        if noise is None:
            return sample
        return self.perturb(sample, noise, sigma)
