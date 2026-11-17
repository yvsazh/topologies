import math

import torch
from torch import nn


class SmoothGrad(nn.Module):
    component_id = "smoothgrad"
    PRESETS = {
        "tiny": {"sigma": 0.1, "n_samples": 2, "seed": 0},
        "small": {"sigma": 0.1, "n_samples": 16, "seed": 0},
        "base": {"sigma": 0.1, "n_samples": 50, "seed": 0},
    }

    def __init__(self, sigma=0.1, n_samples=50, seed=0):
        super().__init__()
        if isinstance(sigma, bool) or not isinstance(sigma, (int, float)) or not math.isfinite(sigma) or sigma < 0:
            raise ValueError("sigma must be a finite non-negative number")
        if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int):
            error_type = ValueError
            raise error_type("seed must be an integer")
        self.sigma = float(sigma)
        self.n_samples = n_samples
        self.seed = seed

    def forward(self, inputs, score_fn=None, target=None):
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("inputs must be a torch tensor")
        if inputs.ndim < 2 or inputs.shape[0] < 1 or inputs[0].numel() < 1:
            raise ValueError("inputs must have a non-empty batch and feature dimensions")
        if not inputs.is_floating_point():
            raise ValueError("inputs must be a floating-point tensor")
        if score_fn is None:
            if target is not None:
                raise ValueError("target can only be used with class scores from score_fn")
            use_sum = False
            use_quadratic = True
        elif isinstance(score_fn, str):
            if score_fn != "sum":
                raise ValueError("the only supported score selector is 'sum'")
            if target is not None:
                raise ValueError("target can only be used with class scores from score_fn")
            use_sum = True
            use_quadratic = False
        elif callable(score_fn):
            use_sum = False
            use_quadratic = False
        else:
            raise TypeError("score_fn must be callable or the 'sum' selector")

        batch_size = inputs.shape[0]
        feature_shape = tuple(inputs.shape[1:])
        range_per_example = inputs.flatten(start_dim=1).amax(dim=1) - inputs.flatten(start_dim=1).amin(dim=1)
        noise_shape = (self.n_samples,) + tuple(inputs.shape)
        generator = torch.Generator(device=inputs.device).manual_seed(self.seed)

        with torch.enable_grad():
            noise = torch.randn(noise_shape, dtype=inputs.dtype, device=inputs.device, generator=generator)
            scale_shape = (1, batch_size) + (1,) * len(feature_shape)
            noise = noise * (self.sigma * range_per_example).reshape(scale_shape)
            noisy_inputs = (inputs.unsqueeze(0) + noise).reshape((-1,) + feature_shape)
            noisy_inputs.requires_grad_(True)
            if use_sum:
                scores = noisy_inputs.flatten(start_dim=1).sum(dim=-1)
            elif use_quadratic:
                scores = noisy_inputs.flatten(start_dim=1).square().sum(dim=-1)
            else:
                scores = score_fn(noisy_inputs)
            if not isinstance(scores, torch.Tensor):
                raise TypeError("score_fn must return a torch tensor")
            if scores.shape not in ((self.n_samples * batch_size,), (self.n_samples * batch_size, 1)) and not (
                scores.ndim == 2 and scores.shape[0] == self.n_samples * batch_size
            ):
                raise ValueError("score_fn must return one score or a class-score row per example")
            if not scores.is_floating_point() or scores.device != inputs.device:
                raise ValueError("score_fn output must be floating-point and on the input device")
            if not scores.requires_grad:
                raise ValueError("score_fn output must be differentiable with respect to inputs")
            if scores.ndim == 2:
                if target is None:
                    scores = scores.sum(dim=-1)
                else:
                    if isinstance(target, bool) or not isinstance(target, int):
                        raise ValueError("target must be an integer class index")
                    if target < 0 or target >= scores.shape[1]:
                        raise ValueError("target is outside the available class scores")
                    scores = scores[:, target]
            elif target is not None:
                raise ValueError("target requires class scores with shape [batch, classes]")
            gradients = torch.autograd.grad(scores.sum(), noisy_inputs, create_graph=True)[0]
            gradients = gradients.reshape((self.n_samples, batch_size) + feature_shape)
            return gradients.mean(dim=0)
