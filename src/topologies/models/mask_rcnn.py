import torch
from torch import nn


class MaskRCNN(nn.Module):
    architecture_id = "mask-rcnn"
    def __init__(self, in_channels=256, num_classes=81, hidden_channels=256, num_convs=4):
        super().__init__()
        if in_channels <= 0 or num_classes <= 0 or hidden_channels <= 0 or num_convs <= 0:
            raise ValueError("channel counts, class count, and convolution count must be positive")
        layers = []
        channels = in_channels
        for _ in range(num_convs):
            layers.extend((nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True)))
            channels = hidden_channels
        self.mask_convs = nn.Sequential(*layers)
        self.mask_upsample = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=2, stride=2)
        self.mask_logits = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)

    def forward(self, pooled_regions):
        if pooled_regions.ndim != 4:
            raise ValueError("pooled_regions must have shape (regions, channels, height, width)")
        if pooled_regions.shape[1] != self.mask_convs[0].in_channels:
            raise ValueError("pooled_regions channel count does not match in_channels")
        features = self.mask_convs(pooled_regions)
        return self.mask_logits(torch.relu(self.mask_upsample(features)))
