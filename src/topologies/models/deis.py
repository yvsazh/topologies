import torch
from torch import nn


class DEIS(nn.Module):
    architecture_id = "deis"
    def __init__(self, features=4):
        super().__init__()
        if features < 1:
            raise ValueError("features must be positive")
        self.features = features

    def forward(self, sample, epsilon, alpha_t, sigma_t, alpha_s, sigma_s):
        if sample.shape != epsilon.shape:
            raise ValueError("sample and epsilon must have matching shapes")
        if sample.ndim == 0 or sample.shape[-1] != self.features:
            raise ValueError("last input dimension must equal features")
        alpha_t = torch.as_tensor(alpha_t, dtype=sample.dtype, device=sample.device)
        sigma_t = torch.as_tensor(sigma_t, dtype=sample.dtype, device=sample.device)
        alpha_s = torch.as_tensor(alpha_s, dtype=sample.dtype, device=sample.device)
        sigma_s = torch.as_tensor(sigma_s, dtype=sample.dtype, device=sample.device)
        if torch.any(alpha_t == 0):
            raise ValueError("alpha_t must be nonzero")
        ratio = alpha_s / alpha_t
        return ratio * sample + (sigma_s - ratio * sigma_t) * epsilon
