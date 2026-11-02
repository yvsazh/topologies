import torch
from torch import nn


class Cutout(nn.Module):
    component_id = "cutout"

    def __init__(self, n_holes=1, length=16):
        super().__init__()
        if not isinstance(n_holes, int) or isinstance(n_holes, bool) or n_holes <= 0:
            raise ValueError("n_holes must be a positive integer")
        if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
            raise ValueError("length must be a positive integer")
        self.n_holes = n_holes
        self.length = length

    def forward(self, x):
        if not torch.is_tensor(x):
            raise ValueError("input must be a tensor with CHW or NCHW shape")
        if x.ndim == 3:
            images = x.unsqueeze(0)
            squeeze_batch = True
        elif x.ndim == 4:
            images = x
            squeeze_batch = False
        else:
            raise ValueError("input must have CHW or NCHW shape")
        if any(size <= 0 for size in images.shape):
            raise ValueError("input dimensions must be positive")
        if not self.training:
            return x

        batch_size, _, height, width = images.shape
        rows = torch.randint(0, height, (batch_size, self.n_holes), device=x.device)
        columns = torch.randint(0, width, (batch_size, self.n_holes), device=x.device)
        row_positions = torch.arange(height, device=x.device).view(1, height, 1)
        column_positions = torch.arange(width, device=x.device).view(1, 1, width)
        mask = torch.ones((batch_size, height, width), dtype=torch.bool, device=x.device)
        offset = self.length // 2

        for hole in range(self.n_holes):
            row_start = (rows[:, hole] - offset).view(batch_size, 1, 1)
            column_start = (columns[:, hole] - offset).view(batch_size, 1, 1)
            covered = (
                (row_positions >= row_start)
                & (row_positions < row_start + self.length)
                & (column_positions >= column_start)
                & (column_positions < column_start + self.length)
            )
            mask &= ~covered

        output = images * mask.unsqueeze(1)
        if squeeze_batch:
            return output.squeeze(0)
        return output
