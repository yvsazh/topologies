import torch


class Snake(torch.nn.Module):
    component_id = "snake"
    def __init__(self, frequency=1.0):
        if frequency <= 0:
            raise ValueError("frequency must be positive")
        super().__init__()
        self.frequency = frequency

    def forward(self, x):
        return x + torch.sin(self.frequency * x).square() / self.frequency
