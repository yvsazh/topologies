import torch
from torch import nn


class CenterNetDenseHead(nn.Module):
    architecture_id = "centernet"

    def __init__(self, in_channels=3, num_classes=80, hidden_channels=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.heatmap_head = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)
        self.offset_head = nn.Conv2d(hidden_channels, 2, kernel_size=1)
        self.size_head = nn.Conv2d(hidden_channels, 2, kernel_size=1)

    def forward(self, x):
        features = self.backbone(x)
        return {
            "heatmap": torch.sigmoid(self.heatmap_head(features)),
            "offset": self.offset_head(features),
            "size": self.size_head(features),
        }
