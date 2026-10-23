import torch
from torch import nn


class GradCAM(nn.Module):
    component_id = "grad-cam"

    def __init__(self):
        super().__init__()

    def forward(self, activations, gradients):
        if not torch.is_tensor(activations) or not torch.is_tensor(gradients):
            raise ValueError("activations and gradients must be tensors")
        if activations.ndim != 4 or gradients.ndim != 4:
            raise ValueError("activations and gradients must have rank 4 with shape [N, K, H, W]")
        if activations.shape != gradients.shape:
            raise ValueError("activations and gradients must have identical shapes")
        if any(size < 1 for size in activations.shape):
            raise ValueError("activations and gradients must have nonempty dimensions")
        if not activations.is_floating_point() or not gradients.is_floating_point():
            raise ValueError("activations and gradients must have floating-point dtypes")
        if activations.device != gradients.device or activations.dtype != gradients.dtype:
            raise ValueError("activations and gradients must share a device and dtype")
        channel_weights = gradients.mean(dim=(-2, -1))
        weighted_activations = activations * channel_weights[:, :, None, None]
        return torch.relu(weighted_activations.sum(dim=1))
