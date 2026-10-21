import math

import torch
from torch import nn
from torch.nn import functional as F


def noise_factor(value):
    return value.sign() * value.abs().sqrt()


class NoisyNets(nn.Module):
    component_id = "noisy-nets"
    def __init__(self, in_features=4, out_features=3, bias=True, sigma_init=0.5):
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("feature sizes must be positive")
        if sigma_init < 0:
            raise ValueError("sigma_init must be non-negative")
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features)) if bias else None
        self.bias_sigma = nn.Parameter(torch.empty(out_features)) if bias else None
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.noise_enabled = True
        self.reset_parameters(sigma_init)
        self.reset_noise()

    def reset_parameters(self, sigma_init):
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, sigma_init / math.sqrt(self.in_features))
        if self.bias_mu is not None:
            nn.init.uniform_(self.bias_mu, -bound, bound)
            nn.init.constant_(self.bias_sigma, sigma_init / math.sqrt(self.out_features))

    @torch.no_grad()
    def reset_noise(self):
        input_noise = noise_factor(torch.randn(self.in_features, device=self.weight_mu.device, dtype=self.weight_mu.dtype))
        output_noise = noise_factor(torch.randn(self.out_features, device=self.weight_mu.device, dtype=self.weight_mu.dtype))
        self.weight_epsilon.copy_(output_noise.outer(input_noise))
        self.bias_epsilon.copy_(output_noise)

    def forward(self, inputs):
        if inputs.shape[-1] != self.in_features:
            raise ValueError("last input dimension must equal in_features")
        if self.noise_enabled:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon if self.bias_mu is not None else None
        else:
            weight, bias = self.weight_mu, self.bias_mu
        return F.linear(inputs, weight, bias)
