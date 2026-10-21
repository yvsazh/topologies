import torch
from torch import nn


class Pix2Pix(nn.Module):
    architecture_id = "pix2pix"
    def __init__(self, channels=1, base_channels=32, depth=3):
        super().__init__()
        if channels < 1 or base_channels < 1 or depth < 1:
            raise ValueError("channels, base_channels, and depth must be positive")
        self.channels = channels
        encoder = []
        widths = [channels]
        for level in range(depth):
            width = min(base_channels * (2 ** level), base_channels * 8)
            encoder.extend([
                nn.Conv2d(widths[-1], width, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            widths.append(width)
        self.encoder = nn.ModuleList()
        cursor = 0
        for level in range(depth):
            self.encoder.append(nn.Sequential(*encoder[cursor:cursor + 2]))
            cursor += 2
        self.decoder = nn.ModuleList()
        current_width = widths[-1]
        for level in reversed(range(depth)):
            out_width = widths[level]
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose2d(current_width, out_width, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ))
            current_width = out_width + (widths[level] if level > 0 else 0)
        self.generator_head = nn.Sequential(nn.Conv2d(widths[0], channels, 3, padding=1), nn.Tanh())
        discriminator_width = max(base_channels, 4)
        self.discriminator = nn.Sequential(
            nn.Conv2d(channels * 2, discriminator_width, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(discriminator_width, discriminator_width * 2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(discriminator_width * 2, 1, 3, padding=1),
        )

    def generate(self, condition):
        features = []
        value = condition
        for layer in self.encoder:
            value = layer(value)
            features.append(value)
        for level, decoder_layer in zip(reversed(range(len(self.decoder))), self.decoder):
            value = decoder_layer(value)
            if level > 0:
                skip = features[level - 1]
                if value.shape[-2:] != skip.shape[-2:]:
                    value = nn.functional.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                value = torch.cat((value, skip), dim=1)
        value = nn.functional.interpolate(value, size=condition.shape[-2:], mode="bilinear", align_corners=False)
        return self.generator_head(value)

    def discriminate(self, condition, generated):
        if condition.shape != generated.shape:
            raise ValueError("condition and generated images must have matching shapes")
        return self.discriminator(torch.cat((condition, generated), dim=1))

    def forward(self, condition):
        generated = self.generate(condition)
        return generated, self.discriminate(condition, generated)
