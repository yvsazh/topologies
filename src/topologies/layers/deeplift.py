import torch
from torch import nn


class DeepLIFT(nn.Module):
    component_id = "deeplift"

    def __init__(self):
        super().__init__()

    @staticmethod
    def _value_error(message: str) -> ValueError:
        return ValueError(message)

    def forward(
        self,
        inputs: torch.Tensor,
        reference: torch.Tensor | None = None,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not isinstance(inputs, torch.Tensor):
            raise self._value_error("inputs must be a torch.Tensor")
        if inputs.ndim < 1:
            raise ValueError("inputs must have at least one dimension")
        if (reference is None) != (weight is None):
            raise ValueError("reference and weight must either both be supplied or both be omitted")
        if reference is None:
            reference = torch.zeros_like(inputs)
            weight = torch.eye(inputs.shape[-1], dtype=inputs.dtype, device=inputs.device)
        tensors = [("inputs", inputs), ("reference", reference), ("weight", weight)]
        if bias is not None:
            tensors.append(("bias", bias))
        for name, tensor in tensors:
            if not isinstance(tensor, torch.Tensor):
                raise self._value_error(f"{name} must be a torch.Tensor")
            if not tensor.is_floating_point():
                raise ValueError(f"{name} must have a floating-point dtype")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} must contain only finite values")
        if weight.ndim != 2:
            raise ValueError("weight must have shape [outputs, inputs]")
        if inputs.shape[-1] != weight.shape[1]:
            raise ValueError("the input dimension must match the last weight dimension")
        if reference.ndim < 1 or reference.shape[-1] != inputs.shape[-1]:
            raise ValueError("reference must have the same final dimension as inputs")
        try:
            broadcast_shape = torch.broadcast_shapes(inputs.shape, reference.shape)
        except RuntimeError as error:
            raise ValueError("reference must be broadcastable to inputs") from error
        if tuple(broadcast_shape) != tuple(inputs.shape):
            raise ValueError("reference must broadcast to the shape of inputs")
        if any(tensor.device != inputs.device for _, tensor in tensors):
            raise ValueError("inputs, reference, weight, and bias must be on the same device")
        if bias is not None and bias.shape != (weight.shape[0],):
            raise ValueError("bias must have shape [outputs]")
        differences = inputs - reference
        return differences.unsqueeze(-2) * weight
