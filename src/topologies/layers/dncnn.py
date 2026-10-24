import torch
from torch import nn


class DnCNN(nn.Module):
    component_id = "dncnn"
    PRESETS = {
        "tiny": {"channels": 1, "features": 8, "depth": 3},
        "small": {"channels": 1, "features": 32, "depth": 5},
        "base": {"channels": 1, "features": 64, "depth": 17},
    }

    def __init__(self, channels=1, features=64, depth=17):
        super().__init__()
        if not isinstance(channels, int) or isinstance(channels, bool) or channels not in (1, 3):
            raise ValueError("channels must be 1 or 3")
        if not isinstance(features, int) or isinstance(features, bool) or features <= 0:
            raise ValueError("features must be a positive integer")
        if not isinstance(depth, int) or isinstance(depth, bool) or depth < 2:
            raise ValueError("depth must be an integer greater than or equal to 2")
        self.channels = channels
        self.features = features
        self.depth = depth
        layers = [
            nn.Conv2d(channels, features, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=False),
        ]
        for _ in range(depth - 2):
            layers.extend(
                [
                    nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(features, eps=1e-5, momentum=0.1),
                    nn.ReLU(inplace=False),
                ]
            )
        layers.append(nn.Conv2d(features, channels, kernel_size=3, stride=1, padding=1, bias=True))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 4:
            raise ValueError("input must be a tensor with shape [batch, channels, height, width]")
        if x.shape[1] != self.channels:
            raise ValueError(f"input channel count must equal channels ({self.channels})")
        if any(size <= 0 for size in x.shape):
            raise ValueError("input dimensions must be positive")
        if not x.is_floating_point():
            raise ValueError("input must have a floating point dtype")
        return self.network(x)

    @staticmethod
    def reconstruct(noisy, predicted_residual):
        if not isinstance(noisy, torch.Tensor) or not isinstance(predicted_residual, torch.Tensor):
            raise TypeError("noisy and predicted_residual must be tensors")
        if noisy.shape != predicted_residual.shape:
            raise ValueError("noisy and predicted_residual must have the same shape")
        if noisy.ndim != 4 or noisy.shape[0] <= 0:
            raise ValueError("noisy and predicted_residual must have image batch shape [batch, channels, height, width]")
        return noisy - predicted_residual

    @staticmethod
    def residual_loss(predicted_residual, noisy, clean):
        tensors = (predicted_residual, noisy, clean)
        if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise TypeError("predicted_residual, noisy, and clean must be tensors")
        if predicted_residual.shape != noisy.shape or noisy.shape != clean.shape:
            raise ValueError("predicted_residual, noisy, and clean must have the same shape")
        if noisy.ndim != 4 or noisy.shape[0] <= 0:
            raise ValueError("residual loss inputs must have image batch shape [batch, channels, height, width]")
        target_residual = noisy - clean
        squared_error = (predicted_residual - target_residual).square().flatten(start_dim=1).sum(dim=1)
        return squared_error.sum() / (2 * noisy.shape[0])
