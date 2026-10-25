import torch
from torch import nn


class SDS(nn.Module):
    component_id = "sds"

    def __init__(self):
        super().__init__()

    def forward(self, predicted_noise: torch.Tensor, injected_noise: torch.Tensor) -> torch.Tensor:
        if not isinstance(predicted_noise, torch.Tensor):
            raise TypeError("predicted_noise must be a torch.Tensor")
        if not isinstance(injected_noise, torch.Tensor):
            raise TypeError("injected_noise must be a torch.Tensor")
        if predicted_noise.shape != injected_noise.shape:
            raise ValueError("predicted_noise and injected_noise must have the same shape")
        return predicted_noise - injected_noise
