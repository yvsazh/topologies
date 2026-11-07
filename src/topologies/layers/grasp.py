import torch
from torch import nn


class GraSPScore(nn.Module):
    component_id = "grasp"

    def __init__(self):
        super().__init__()

    def forward(self, weight, hessian_grad):
        if not isinstance(weight, torch.Tensor) or not isinstance(hessian_grad, torch.Tensor):
            raise TypeError("weight and hessian_grad must be tensors")
        if weight.shape != hessian_grad.shape:
            raise ValueError("weight and hessian_grad must have the same shape")
        return -weight * hessian_grad
