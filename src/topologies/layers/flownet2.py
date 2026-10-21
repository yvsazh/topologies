import torch
from torch import nn
from torch.nn import functional as F


class Correlation(nn.Module):
    def __init__(self, max_displacement=2, stride=1):
        super().__init__()
        if max_displacement < 0 or stride < 1:
            raise ValueError("max_displacement must be nonnegative and stride must be positive")
        self.max_displacement = int(max_displacement)
        self.stride = int(stride)

    def forward(self, first, second):
        if first.ndim != 4 or second.ndim != 4:
            raise ValueError("feature maps must have shape (batch, channels, height, width)")
        if first.shape != second.shape:
            raise ValueError("feature maps must have identical shapes")
        radius = self.max_displacement
        padded = F.pad(second, (radius, radius, radius, radius))
        correlations = []
        for dy in range(-radius, radius + 1, self.stride):
            for dx in range(-radius, radius + 1, self.stride):
                shifted = padded[:, :, radius + dy:radius + dy + first.shape[2], radius + dx:radius + dx + first.shape[3]]
                correlations.append((first * shifted).sum(dim=1, keepdim=True))
        return torch.cat(correlations, dim=1)


class FlowNet2(nn.Module):
    component_id = "flownet2"
    def __init__(self, in_channels=3, feature_channels=16, max_displacement=2, hidden_channels=32, stride=1):
        super().__init__()
        if in_channels < 1 or feature_channels < 1 or hidden_channels < 1:
            raise ValueError("channel counts must be positive")
        self.correlation = Correlation(max_displacement=max_displacement, stride=stride)
        volume_channels = ((2 * self.correlation.max_displacement) // self.correlation.stride + 1) ** 2
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.flow = nn.Sequential(
            nn.Conv2d(feature_channels * 2 + volume_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 2, kernel_size=3, padding=1),
        )

    def forward(self, first, second):
        first_features = self.encoder(first)
        second_features = self.encoder(second)
        cost_volume = self.correlation(first_features, second_features)
        return self.flow(torch.cat((first_features, second_features, cost_volume), dim=1))
