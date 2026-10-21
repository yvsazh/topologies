import torch
from torch import nn


class MoiraiPatchProjection(nn.Module):
    component_id = "moirai"

    def __init__(self, patch_size=4, d_model=32):
        super().__init__()
        if patch_size < 1 or d_model < 1:
            raise ValueError("patch_size and d_model must be positive")
        self.patch_size = patch_size
        self.patch_projection = nn.Linear(patch_size, d_model)

    def forward(self, values):
        if not torch.is_tensor(values):
            values = torch.as_tensor(values, dtype=self.patch_projection.weight.dtype, device=self.patch_projection.weight.device)
        if values.ndim != 3:
            raise ValueError("values must have shape (batch, time, variates)")
        batch, time, variates = values.shape
        if time < self.patch_size or time % self.patch_size:
            raise ValueError("time dimension must be a positive multiple of patch_size")
        patches = values.transpose(1, 2).reshape(batch, variates, time // self.patch_size, self.patch_size)
        return self.patch_projection(patches)
