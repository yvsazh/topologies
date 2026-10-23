import torch
from torch import nn


class TCAVConceptScore(nn.Module):
    component_id = "tcav"
    def __init__(self, *args, **kwargs):
        if args or kwargs:
            raise ValueError("TCAVConceptScore takes no constructor arguments")
        super().__init__()

    def directional_sensitivities(self, activation_gradients, cav):
        if not isinstance(activation_gradients, torch.Tensor):
            raise TypeError("activation_gradients must be a torch.Tensor")
        if not isinstance(cav, torch.Tensor):
            raise TypeError("cav must be a torch.Tensor")
        if activation_gradients.ndim < 2:
            raise ValueError("activation_gradients must include a batch and feature dimensions")
        if activation_gradients.shape[0] == 0:
            raise ValueError("activation_gradients batch must be non-empty")
        if any(size == 0 for size in activation_gradients.shape[1:]):
            raise ValueError("activation_gradients feature dimensions must be non-empty")
        if not torch.is_floating_point(activation_gradients):
            raise ValueError("activation_gradients must have a floating-point dtype")
        if not torch.is_floating_point(cav):
            raise ValueError("cav must have a floating-point dtype")
        if activation_gradients.device != cav.device:
            raise ValueError("activation_gradients and cav must be on the same device")

        flattened_gradients = activation_gradients.reshape(activation_gradients.shape[0], -1)
        if tuple(cav.shape) == tuple(activation_gradients.shape[1:]):
            flattened_cav = cav.reshape(-1)
            cav_norm = torch.linalg.vector_norm(flattened_cav)
            if cav_norm.item() == 0:
                raise ValueError("cav must have a non-zero L2 norm")
            normalized_cav = flattened_cav / cav_norm
            return (flattened_gradients * normalized_cav).sum(dim=-1)
        if tuple(cav.shape) == tuple(activation_gradients.shape):
            flattened_cav = cav.reshape(activation_gradients.shape[0], -1)
            cav_norm = torch.linalg.vector_norm(flattened_cav, dim=-1, keepdim=True)
            if torch.any(cav_norm == 0).item():
                raise ValueError("each per-example cav must have a non-zero L2 norm")
            normalized_cav = flattened_cav / cav_norm
            return (flattened_gradients * normalized_cav).sum(dim=-1)
        raise ValueError(
            "cav must have the same feature shape as activation_gradients or match its full shape"
        )

    def forward(self, activation_gradients, cav, return_sensitivities=False):
        if not isinstance(return_sensitivities, bool):
            raise TypeError("return_sensitivities must be a bool")
        sensitivities = self.directional_sensitivities(activation_gradients, cav)
        if return_sensitivities:
            return sensitivities
        positives = (sensitivities > 0).to(dtype=sensitivities.dtype)
        zero_gradient = activation_gradients.reshape(-1)[0] * 0.0
        return positives.mean() + zero_gradient
