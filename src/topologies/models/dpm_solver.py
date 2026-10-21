import torch
from torch import nn


def dpm_solver_first_order(sample, epsilon, alpha_source, sigma_source, alpha_target, sigma_target):
    alpha_source = torch.as_tensor(alpha_source, dtype=sample.dtype, device=sample.device)
    sigma_source = torch.as_tensor(sigma_source, dtype=sample.dtype, device=sample.device)
    alpha_target = torch.as_tensor(alpha_target, dtype=sample.dtype, device=sample.device)
    sigma_target = torch.as_tensor(sigma_target, dtype=sample.dtype, device=sample.device)
    if any(torch.any(value <= 0) for value in (alpha_source, sigma_source, alpha_target, sigma_target)):
        raise ValueError("alpha and sigma schedule values must be positive")
    log_snr_source = torch.log(alpha_source) - torch.log(sigma_source)
    log_snr_target = torch.log(alpha_target) - torch.log(sigma_target)
    step = log_snr_target - log_snr_source
    return (sigma_target / sigma_source) * sample - alpha_target * torch.expm1(-step) * epsilon


class DPMSolver(nn.Module):
    architecture_id = "dpm-solver"
    def __init__(self, features):
        super().__init__()
        if features <= 0:
            raise ValueError("features must be positive")
        self.features = features

    def forward(self, sample, epsilon, alpha_source, sigma_source, alpha_target, sigma_target):
        if sample.shape != epsilon.shape:
            raise ValueError("sample and epsilon must have matching shapes")
        if sample.ndim == 0 or sample.shape[-1] != self.features:
            raise ValueError("last input dimension must match features")
        return dpm_solver_first_order(sample, epsilon, alpha_source, sigma_source, alpha_target, sigma_target)
