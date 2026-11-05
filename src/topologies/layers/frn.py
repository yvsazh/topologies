import math

import torch
from torch import nn


class FRN(nn.Module):
    component_id = "frn"

    def __init__(self, num_features, epsilon=1e-6, learnable_epsilon=False):
        super().__init__()
        if type(num_features) is not int or num_features <= 0:
            raise ValueError("num_features must be a positive integer")
        if type(epsilon) not in (int, float):
            raise ValueError("epsilon must be a finite positive number")
        try:
            epsilon = float(epsilon)
        except (OverflowError, ValueError):
            raise ValueError("epsilon must be a finite positive number") from None
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be a finite positive number")
        if type(learnable_epsilon) is not bool:
            raise ValueError("learnable_epsilon must be a boolean")

        self.num_features = num_features
        self.epsilon = float(epsilon)
        self.learnable_epsilon = learnable_epsilon
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.tau = nn.Parameter(torch.zeros(num_features))
        if learnable_epsilon:
            self.epsilon_raw = nn.Parameter(torch.tensor(1e-4 - 1e-6))
        else:
            self.register_parameter("epsilon_raw", None)

    def forward(self, x):
        if not torch.is_tensor(x):
            raise ValueError("input must be a torch.Tensor")
        if x.ndim not in (3, 4):
            raise ValueError(
                "input must have 3 dimensions [channels, height, width] or 4 dimensions [batch, channels, height, width]"
            )
        if any(size == 0 for size in x.shape):
            raise ValueError("input dimensions must be nonempty")
        channel_axis = x.ndim - 3
        if x.shape[channel_axis] != self.num_features:
            raise ValueError("input channel count must match num_features for CHW or NCHW input")
        if not x.is_floating_point():
            raise ValueError("input must have a floating-point dtype")

        if self.learnable_epsilon:
            epsilon = 1e-6 + self.epsilon_raw.abs()
        else:
            epsilon = self.epsilon
        nu2 = x.square().mean(dim=(-2, -1), keepdim=True)
        normalized = x * torch.rsqrt(nu2 + epsilon)
        parameter_shape = [1] * x.ndim
        parameter_shape[channel_axis] = self.num_features
        gamma = self.gamma.view(parameter_shape)
        beta = self.beta.view(parameter_shape)
        tau = self.tau.view(parameter_shape)
        affine = gamma * normalized + beta
        return torch.where(affine >= tau, affine, tau)
