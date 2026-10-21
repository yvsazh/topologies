import torch
from torch import nn


class VPred(nn.Module):
    architecture_id = "v-pred"
    def __init__(self, features=4):
        super().__init__()
        if not isinstance(features, int) or features <= 0:
            raise ValueError("features must be a positive integer")
        self.features = features

    def forward(self, clean, noise, alpha, sigma):
        if clean.shape != noise.shape:
            raise ValueError("clean and noise must have matching shapes")
        if clean.shape[-1] != self.features:
            raise ValueError("last dimension must match features")
        return alpha * noise - sigma * clean

    @staticmethod
    def noise_from_velocity(noisy_sample, velocity, alpha, sigma):
        if noisy_sample.shape != velocity.shape:
            raise ValueError("noisy_sample and velocity must have matching shapes")
        return sigma * noisy_sample + alpha * velocity
