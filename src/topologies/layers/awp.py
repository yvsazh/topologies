import math

import torch


class AdversarialWeightPerturbation(torch.nn.Module):
    component_id = "awp"

    def __init__(self, gamma=0.01):
        super().__init__()
        gamma = float(gamma)
        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be finite and nonnegative")
        self.gamma = gamma

    def forward(self, parameter, gradient, perturbation=None, step_size=1.0):
        if parameter.shape != gradient.shape:
            raise ValueError("parameter and gradient must have identical shapes")
        if not parameter.is_floating_point():
            raise ValueError("parameter must have a floating-point dtype")
        step_size = float(step_size)
        if not math.isfinite(step_size) or step_size < 0:
            raise ValueError("step_size must be finite and nonnegative")

        gradient = gradient.to(device=parameter.device, dtype=parameter.dtype)
        if perturbation is None:
            perturbation = torch.zeros_like(parameter)
        elif perturbation.shape != parameter.shape:
            raise ValueError("perturbation and parameter must have identical shapes")
        else:
            perturbation = perturbation.to(device=parameter.device, dtype=parameter.dtype)

        parameter_norm = torch.linalg.vector_norm(parameter)
        radius = self.gamma * parameter_norm
        if radius.item() == 0:
            return parameter
        gradient_norm = torch.linalg.vector_norm(gradient)
        proposal = perturbation
        if gradient_norm.item() != 0:
            proposal = proposal + step_size * parameter_norm * gradient / gradient_norm
        proposal_norm = torch.linalg.vector_norm(proposal)
        if proposal_norm.item() == 0:
            return parameter
        scale = torch.clamp(radius / proposal_norm, max=1.0)
        return parameter + proposal * scale

    @staticmethod
    def training_update(parameter, perturbed_gradient, learning_rate):
        if parameter.shape != perturbed_gradient.shape:
            raise ValueError("parameter and perturbed_gradient must have identical shapes")
        learning_rate = float(learning_rate)
        if not math.isfinite(learning_rate) or learning_rate < 0:
            raise ValueError("learning_rate must be finite and nonnegative")
        gradient = perturbed_gradient.to(device=parameter.device, dtype=parameter.dtype)
        return parameter - learning_rate * gradient
