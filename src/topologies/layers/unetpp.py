import torch
from torch import nn
from torch.nn import functional as F


class UNetPlusPlus(nn.Module):
    architecture_id = "unetpp"
    component_id = "unetpp"
    def __init__(self, in_channels=1, hidden_channels=32, num_classes=2):
        super().__init__()
        self.encoder0 = self._block(in_channels, hidden_channels)
        self.encoder1 = self._block(hidden_channels, hidden_channels * 2)
        self.encoder2 = self._block(hidden_channels * 2, hidden_channels * 4)
        self.nested0_1 = self._block(hidden_channels * 3, hidden_channels)
        self.nested1_1 = self._block(hidden_channels * 6, hidden_channels * 2)
        self.nested0_2 = self._block(hidden_channels * 4, hidden_channels)
        self.classifier = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)

    @staticmethod
    def _block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, image):
        encoder0 = self.encoder0(image)
        encoder1 = self.encoder1(F.max_pool2d(encoder0, kernel_size=2))
        encoder2 = self.encoder2(F.max_pool2d(encoder1, kernel_size=2))
        upsampled1 = F.interpolate(encoder1, size=encoder0.shape[-2:], mode="bilinear", align_corners=False)
        nested0_1 = self.nested0_1(torch.cat((encoder0, upsampled1), dim=1))
        upsampled2 = F.interpolate(encoder2, size=encoder1.shape[-2:], mode="bilinear", align_corners=False)
        nested1_1 = self.nested1_1(torch.cat((encoder1, upsampled2), dim=1))
        upsampled_nested1 = F.interpolate(
            nested1_1, size=encoder0.shape[-2:], mode="bilinear", align_corners=False
        )
        nested0_2 = self.nested0_2(torch.cat((encoder0, nested0_1, upsampled_nested1), dim=1))
        return self.classifier(nested0_2)
