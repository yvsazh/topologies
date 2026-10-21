import torch
from torch import nn


class UniPC(nn.Module):
    architecture_id = "unipc"
    def __init__(self, features=4):
        super().__init__()
        if features < 1:
            raise ValueError("features must be positive")
        self.features = features

    def predictor(self, sample, epsilon, alpha_t, sigma_t, alpha_s, sigma_s):
        self._validate(sample, epsilon)
        alpha_t, sigma_t, alpha_s, sigma_s = self._coefficients(
            sample, alpha_t, sigma_t, alpha_s, sigma_s
        )
        h = torch.log(alpha_s / sigma_s) - torch.log(alpha_t / sigma_t)
        return (alpha_s / alpha_t) * sample - sigma_s * torch.expm1(h) * epsilon

    def corrector(self, sample, predictor_epsilon, corrector_epsilon, alpha_t, sigma_t, alpha_s, sigma_s):
        self._validate(sample, predictor_epsilon)
        self._validate(sample, corrector_epsilon)
        alpha_t, sigma_t, alpha_s, sigma_s = self._coefficients(
            sample, alpha_t, sigma_t, alpha_s, sigma_s
        )
        h = torch.log(alpha_s / sigma_s) - torch.log(alpha_t / sigma_t)
        phi_2 = self._phi_2(h)
        base = (alpha_s / alpha_t) * sample
        return base - sigma_s * (
            torch.expm1(h) * predictor_epsilon
            + phi_2 * (corrector_epsilon - predictor_epsilon)
        )

    def forward(self, sample, epsilon, corrector_epsilon, alpha_t, sigma_t, alpha_s, sigma_s):
        return self.corrector(
            sample, epsilon, corrector_epsilon, alpha_t, sigma_t, alpha_s, sigma_s
        )

    @staticmethod
    def _phi_2(h):
        small = h.abs() < 1e-4
        safe_h = torch.where(small, torch.ones_like(h), h)
        exact = torch.expm1(h) / safe_h - 1
        series = h / 2 + h.square() / 6 + h.pow(3) / 24 + h.pow(4) / 120
        return torch.where(small, series, exact)

    def _validate(self, sample, epsilon):
        if sample.shape != epsilon.shape:
            raise ValueError("sample and epsilon must have matching shapes")
        if sample.ndim == 0 or sample.shape[-1] != self.features:
            raise ValueError("last input dimension must equal features")

    @staticmethod
    def _coefficients(sample, alpha_t, sigma_t, alpha_s, sigma_s):
        values = tuple(
            torch.as_tensor(value, dtype=sample.dtype, device=sample.device)
            for value in (alpha_t, sigma_t, alpha_s, sigma_s)
        )
        if any(torch.any(~torch.isfinite(value)) or torch.any(value <= 0) for value in values):
            raise ValueError("alpha and sigma coefficients must be finite and positive")
        return values
