import torch.nn as nn


class ReversibleResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.f = nn.Linear(dim, dim, bias=True)
        self.g = nn.Linear(dim, dim, bias=True)

    def forward(self, x1, x2):
        y1 = x1 + self.f(x2)
        y2 = x2 + self.g(y1)
        return y1, y2

    def reconstruct(self, y1, y2):
        x2 = y2 - self.g(y1)
        x1 = y1 - self.f(x2)
        return x1, x2
