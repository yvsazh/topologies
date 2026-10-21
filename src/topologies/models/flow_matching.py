import torch
from torch import nn


def straight_path(noise, data, time):
    return (1 - time) * noise + time * data


def target_velocity(noise, data):
    return data - noise


def flow_matching_loss(prediction, noise, data):
    return torch.mean((prediction - target_velocity(noise, data)) ** 2)


class FlowMatching(nn.Module):
    architecture_id = "flow-matching"
    def __init__(self, features=4):
        super().__init__()
        if features <= 0:
            raise ValueError("features must be positive")
        self.features = features

    def forward(self, noise, data, time):
        if noise.shape != data.shape or noise.shape[-1] != self.features:
            raise ValueError("noise and data must have matching shapes ending in features")
        return straight_path(noise, data, time)

    def target(self, noise, data):
        if noise.shape != data.shape or noise.shape[-1] != self.features:
            raise ValueError("noise and data must have matching shapes ending in features")
        return target_velocity(noise, data)

    def loss(self, prediction, noise, data):
        return flow_matching_loss(prediction, noise, data)
