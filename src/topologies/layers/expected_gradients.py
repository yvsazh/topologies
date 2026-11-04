import torch
from torch import nn


class ExpectedGradients(nn.Module):
    component_id = "expected-gradients"

    def __init__(self, samples=1):
        super().__init__()
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
            raise ValueError("samples must be a positive integer")
        self.samples = samples

    def forward(self, function, inputs, references):
        if isinstance(function, str):
            if function != "sum":
                raise ValueError("the only supported function selector is 'sum'")
            use_sum = True
        elif callable(function):
            use_sum = False
        else:
            raise TypeError("function must be callable or the 'sum' selector")
        if not isinstance(inputs, torch.Tensor) or not isinstance(references, torch.Tensor):
            raise TypeError("inputs and references must be torch tensors")
        if inputs.ndim < 2 or references.ndim != inputs.ndim:
            raise ValueError("inputs and references must have matching batch and feature dimensions")
        if inputs.shape[0] < 1 or references.shape[0] < 1:
            raise ValueError("inputs and references must contain at least one example")
        if tuple(inputs.shape[1:]) != tuple(references.shape[1:]):
            raise ValueError("inputs and references must have matching feature dimensions")
        if not inputs.is_floating_point() or not references.is_floating_point():
            raise ValueError("inputs and references must be floating-point tensors")
        if inputs.dtype != references.dtype or inputs.device != references.device:
            raise ValueError("inputs and references must have the same dtype and device")

        batch_size = inputs.shape[0]
        reference_count = references.shape[0]
        reference_indices = torch.randint(
            reference_count,
            (batch_size, self.samples),
            device=inputs.device,
        )
        alpha_shape = (batch_size, self.samples) + (1,) * (inputs.ndim - 1)
        alphas = torch.rand(alpha_shape, dtype=inputs.dtype, device=inputs.device)

        with torch.enable_grad():
            sampled_references = references[reference_indices]
            differences = inputs.unsqueeze(1) - sampled_references
            interpolated = sampled_references + alphas * differences
            interpolated = interpolated.reshape((-1,) + tuple(inputs.shape[1:]))
            interpolated.requires_grad_(True)
            if use_sum:
                output = interpolated.flatten(start_dim=1).sum(dim=-1)
            else:
                output = function(interpolated)
            count = batch_size * self.samples
            if not isinstance(output, torch.Tensor):
                raise TypeError("function must return a torch tensor")
            if output.shape not in ((count,), (count, 1)):
                raise ValueError("function must return one scalar per example")
            if not output.is_floating_point() or output.device != inputs.device:
                raise ValueError("function output must be floating-point and on the input device")
            if not output.requires_grad:
                raise ValueError("function output must be differentiable with respect to its input")
            gradients = torch.autograd.grad(output.sum(), interpolated, create_graph=True)[0]
            gradients = gradients.reshape((batch_size, self.samples) + tuple(inputs.shape[1:]))
            return (differences * gradients).mean(dim=1)
