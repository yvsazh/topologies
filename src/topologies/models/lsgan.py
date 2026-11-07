from itertools import chain

import torch
from torch import nn


class LSGAN(nn.Module):
    architecture_id = "lsgan"
    PRESETS = {
        "tiny": {"latent_dim": 8, "image_size": 32, "out_channels": 3, "base_channels": 4},
        "small": {"latent_dim": 64, "image_size": 64, "out_channels": 3, "base_channels": 16},
        "base": {"latent_dim": 1024, "image_size": 112, "out_channels": 3, "base_channels": 64},
    }

    def __init__(self, latent_dim=1024, image_size=112, out_channels=3, base_channels=64):
        super().__init__()
        dimensions = {
            "latent_dim": latent_dim,
            "image_size": image_size,
            "out_channels": out_channels,
            "base_channels": base_channels,
        }
        for name, value in dimensions.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if image_size < 32 or image_size % 16 != 0:
            raise ValueError("image_size must be at least 32 and divisible by 16")

        self.latent_dim = latent_dim
        self.image_size = image_size
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.seed_size = image_size // 16
        projection_channels = base_channels * 4
        self.generator_projection = nn.Linear(
            latent_dim, projection_channels * self.seed_size * self.seed_size
        )
        generator_layers = [nn.BatchNorm2d(projection_channels, eps=1e-5, momentum=0.1)]
        generator_blocks = (
            (base_channels * 2, 2),
            (base_channels, 2),
            (base_channels, 1),
            (base_channels, 1),
            (max(base_channels // 2, 1), 2),
        )
        current_channels = projection_channels
        for output_channels, stride in generator_blocks:
            generator_layers.extend(
                (
                    nn.ConvTranspose2d(
                        current_channels,
                        output_channels,
                        kernel_size=5,
                        stride=stride,
                        padding=2,
                        output_padding=1 if stride == 2 else 0,
                    ),
                    nn.BatchNorm2d(output_channels, eps=1e-5, momentum=0.1),
                    nn.ReLU(inplace=True),
                )
            )
            current_channels = output_channels
        generator_layers.extend(
            (
                nn.ConvTranspose2d(
                    current_channels,
                    out_channels,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    output_padding=1,
                ),
                nn.Tanh(),
            )
        )
        self.generator = nn.Sequential(*generator_layers)

        discriminator_layers = [
            nn.Conv2d(out_channels, base_channels, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        current_channels = base_channels
        for output_channels in (base_channels * 2, base_channels * 4, base_channels * 8):
            discriminator_layers.extend(
                (
                    nn.Conv2d(current_channels, output_channels, kernel_size=5, stride=2, padding=2),
                    nn.BatchNorm2d(output_channels, eps=1e-5, momentum=0.1),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            current_channels = output_channels
        self.discriminator = nn.Sequential(*discriminator_layers)
        self.score_head = nn.Linear(current_channels * self.seed_size * self.seed_size, 1)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def generate(self, latent):
        if not isinstance(latent, torch.Tensor) or latent.ndim != 2:
            raise ValueError(f"latent must have shape [batch, {self.latent_dim}]")
        if latent.shape[0] < 1 or latent.shape[1] != self.latent_dim:
            raise ValueError(f"latent must have shape [batch, {self.latent_dim}] with a non-empty batch")
        projected = self.generator_projection(latent)
        projected = projected.reshape(
            latent.shape[0], self.base_channels * 4, self.seed_size, self.seed_size
        )
        generated = self.generator(projected)
        if generated.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError("latent must produce the configured image size")
        return generated

    def forward(self, latent):
        return self.generate(latent)

    def discriminator_score(self, images):
        expected_shape = (self.out_channels, self.image_size, self.image_size)
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError(
                f"images must have shape [batch, {self.out_channels}, {self.image_size}, {self.image_size}]"
            )
        if images.shape[0] < 1 or tuple(images.shape[1:]) != expected_shape:
            raise ValueError(
                f"images must have shape [batch, {self.out_channels}, {self.image_size}, {self.image_size}]"
            )
        features = self.discriminator(images)
        return self.score_head(features.flatten(start_dim=1))

    @staticmethod
    def _validate_scores(scores, name):
        if not isinstance(scores, torch.Tensor) or scores.ndim != 2:
            raise ValueError(f"{name} must have shape [batch, 1]")
        if scores.shape[0] < 1 or scores.shape[1] != 1:
            raise ValueError(f"{name} must have shape [batch, 1] with a non-empty batch")

    @staticmethod
    def discriminator_loss(real_scores, fake_scores):
        LSGAN._validate_scores(real_scores, "real_scores")
        LSGAN._validate_scores(fake_scores, "fake_scores")
        return 0.5 * (real_scores - 1).square().mean() + 0.5 * fake_scores.square().mean()

    @staticmethod
    def generator_loss(fake_scores):
        LSGAN._validate_scores(fake_scores, "fake_scores")
        return 0.5 * (fake_scores - 1).square().mean()

    def generator_parameters(self):
        return chain(self.generator_projection.parameters(), self.generator.parameters())

    def discriminator_parameters(self):
        return chain(self.discriminator.parameters(), self.score_head.parameters())
