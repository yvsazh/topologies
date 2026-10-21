import torch
from torch import nn


class DQNFamily(nn.Module):
    architecture_id = "dqn-family"
    PRESETS = {"tiny": {"actions": 4}, "small": {"actions": 18}}

    def __init__(self, actions: int):
        super().__init__()
        if isinstance(actions, bool) or not isinstance(actions, int) or actions <= 0:
            raise ValueError("actions must be a positive integer")
        self.actions = actions
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.hidden = nn.Linear(32 * 9 * 9, 256)
        self.output = nn.Linear(256, actions)
        self.activation = nn.ReLU()

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("input must be a torch.Tensor")
        if x.ndim != 4 or tuple(x.shape[1:]) != (4, 84, 84):
            raise ValueError("input must have shape (batch, 4, 84, 84)")
        if x.shape[0] == 0:
            raise ValueError("input batch must not be empty")
        if not x.is_floating_point():
            raise ValueError("input must have a floating-point dtype")
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = self.activation(self.hidden(x))
        return self.output(x)
