import torch
from torch import nn


class ScaffoldCorrection(nn.Module):
    component_id = "scaffold"

    def __init__(self, *args, **kwargs):
        super().__init__()
        if args or kwargs:
            raise ValueError("ScaffoldCorrection does not accept constructor arguments")

    def forward(self, grad, server_control, client_control):
        inputs = (grad, server_control, client_control)
        if not all(isinstance(value, torch.Tensor) for value in inputs):
            raise ValueError("grad, server_control, and client_control must be tensors")
        if grad.shape != server_control.shape or grad.shape != client_control.shape:
            raise ValueError("grad, server_control, and client_control must have matching shapes")
        if grad.numel() == 0:
            raise ValueError("grad, server_control, and client_control must be nonempty")
        return grad - client_control + server_control
