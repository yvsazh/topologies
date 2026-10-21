import torch
from torch import nn


class DDIM(nn.Module):
    component_id = "ddim"
    def __init__(self, features=4):
        super().__init__()
        if features <= 0:
            raise ValueError("features must be positive")
        self.features = features

    def forward(self, noisy_sample, epsilon_prediction, alpha_bar_t, alpha_bar_prev):
        if noisy_sample.shape != epsilon_prediction.shape:
            raise ValueError("noisy sample and epsilon prediction must have matching shapes")
        if noisy_sample.ndim == 0 or noisy_sample.shape[-1] != self.features:
            raise ValueError("last input dimension must match features")
        alpha_bar_t = torch.as_tensor(alpha_bar_t, dtype=noisy_sample.dtype, device=noisy_sample.device)
        alpha_bar_prev = torch.as_tensor(alpha_bar_prev, dtype=noisy_sample.dtype, device=noisy_sample.device)
        predicted_clean = (noisy_sample - torch.sqrt(1.0 - alpha_bar_t) * epsilon_prediction) / torch.sqrt(alpha_bar_t)
        return torch.sqrt(alpha_bar_prev) * predicted_clean + torch.sqrt(1.0 - alpha_bar_prev) * epsilon_prediction
