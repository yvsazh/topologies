import math
import numbers

import torch
from torch import nn


class SHAP(nn.Module):
    component_id = "shap"

    def __init__(self, model=None, n_samples=5, local_smoothing=0.0, seed=0, target=None):
        super().__init__()
        if model is not None and not isinstance(model, nn.Module):
            raise ValueError("model must be a torch.nn.Module or None")
        if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer")
        if (
            isinstance(local_smoothing, bool)
            or not isinstance(local_smoothing, numbers.Real)
            or not math.isfinite(local_smoothing)
            or local_smoothing < 0
        ):
            raise ValueError("local_smoothing must be a finite nonnegative number")
        if type(seed) is not int:
            raise ValueError("seed must be a valid sampling seed")
        if target is not None and (
            isinstance(target, bool) or not isinstance(target, int) or target < 0
        ):
            raise ValueError("target must be a nonnegative integer or None")
        self.model = nn.Identity() if model is None else model
        self.n_samples = n_samples
        self.local_smoothing = float(local_smoothing)
        self.seed = seed
        self.target = target

    def forward(self, inputs, baselines):
        if not isinstance(inputs, torch.Tensor) or not isinstance(baselines, torch.Tensor):
            raise TypeError("inputs and baselines must be torch tensors")
        if inputs.ndim < 2 or baselines.ndim != inputs.ndim:
            raise ValueError("inputs and baselines must have matching feature dimensions")
        if inputs.shape[0] < 1:
            raise ValueError("inputs must contain at least one example")
        if baselines.shape[0] < 1:
            raise ValueError("baselines must contain at least one example")
        if tuple(inputs.shape[1:]) != tuple(baselines.shape[1:]):
            raise ValueError("inputs and baselines must have matching feature dimensions")
        if not inputs.is_floating_point() or not baselines.is_floating_point():
            raise ValueError("inputs and baselines must be floating-point tensors")
        if inputs.dtype != baselines.dtype or inputs.device != baselines.device:
            raise ValueError("inputs and baselines must have the same dtype and device")

        with torch.enable_grad():
            batch_size = inputs.shape[0]
            baseline_count = baselines.shape[0]
            generator = torch.Generator(device=inputs.device).manual_seed(self.seed)
            baseline_indices = torch.randint(
                baseline_count,
                (batch_size, self.n_samples),
                generator=generator,
                device=inputs.device,
            )
            sample_shape = (batch_size, self.n_samples) + (1,) * (inputs.ndim - 1)
            alphas = torch.rand(
                sample_shape,
                dtype=inputs.dtype,
                device=inputs.device,
                generator=generator,
            )
            perturbed_inputs = inputs.unsqueeze(1)
            if self.local_smoothing > 0:
                noise = torch.randn(
                    (batch_size, self.n_samples) + tuple(inputs.shape[1:]),
                    dtype=inputs.dtype,
                    device=inputs.device,
                    generator=generator,
                )
                perturbed_inputs = perturbed_inputs + self.local_smoothing * noise
            differences = perturbed_inputs - baselines[baseline_indices]
            interpolated = baselines[baseline_indices] + alphas * differences
            interpolated = interpolated.reshape((-1,) + tuple(inputs.shape[1:]))
            interpolated.requires_grad_(True)
            output = self.model(interpolated)
            sample_count = batch_size * self.n_samples
            if not isinstance(output, torch.Tensor):
                raise TypeError("model must return a torch tensor")
            if output.device != inputs.device or not output.is_floating_point():
                raise ValueError("model output must be floating-point and on the input device")
            if output.ndim == 1 and output.shape[0] == sample_count:
                if self.target is not None:
                    raise ValueError("target must be None for scalar model outputs")
                selected_output = output
            elif output.ndim == 2 and output.shape[0] == sample_count:
                if output.shape[1] == 1:
                    if self.target is not None:
                        raise ValueError("target must be None for scalar model outputs")
                    selected_output = output[:, 0]
                else:
                    if self.target is None:
                        raise ValueError("target is required for multi-output model outputs")
                    if self.target >= output.shape[1]:
                        raise ValueError("target is out of range for model output")
                    selected_output = output[:, self.target]
            else:
                raise ValueError("model must return shape [B], [B, 1], or [B, C]")
            if not selected_output.requires_grad:
                raise ValueError("model output must be differentiable with respect to its input")
            gradients = torch.autograd.grad(
                selected_output.sum(), interpolated, create_graph=True
            )[0]
            gradients = gradients.reshape(
                (batch_size, self.n_samples) + tuple(inputs.shape[1:])
            )
            return (differences * gradients).mean(dim=1)
