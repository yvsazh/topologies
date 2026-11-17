import torch
from torch import nn


class SEWResNet(nn.Module):
    component_id = "sew-resnet"

    def forward(self, identity, residual):
        if not isinstance(identity, torch.Tensor) or not isinstance(residual, torch.Tensor):
            raise TypeError("identity and residual must be torch tensors")
        if identity.shape != residual.shape:
            raise ValueError(
                f"identity and residual shapes must match; got {tuple(identity.shape)} and {tuple(residual.shape)}"
            )
        return identity + residual
