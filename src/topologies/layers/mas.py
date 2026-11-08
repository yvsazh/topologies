import math
from numbers import Real

import torch
from torch import nn


class MemoryAwareSynapses(nn.Module):
    component_id = "mas"

    def __init__(self, lambda_=1.0):
        super().__init__()
        if isinstance(lambda_, bool) or not isinstance(lambda_, Real) or not math.isfinite(lambda_) or lambda_ < 0:
            raise ValueError("lambda_ must be a finite nonnegative real number")
        self.lambda_ = float(lambda_)

    @staticmethod
    def _validate_vector(value, name, nonnegative=False, allow_batch=False):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        valid_rank = value.ndim == 1 or (allow_batch and value.ndim == 2)
        if not valid_rank or value.numel() == 0 or any(size == 0 for size in value.shape):
            dimensions = "one- or two-dimensional" if allow_batch else "one-dimensional"
            raise ValueError(f"{name} must be a nonempty {dimensions} tensor")
        if not value.is_floating_point():
            raise ValueError(f"{name} must have a floating-point dtype")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"{name} must contain only finite values")
        if nonnegative and torch.any(value < 0).item():
            raise ValueError(f"{name} must contain nonnegative values")

    def estimate_importance(self, model, inputs):
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("inputs must be a tensor")
        if inputs.ndim == 0 or inputs.shape[0] == 0:
            raise ValueError("inputs must have a nonempty batch dimension")
        if not torch.isfinite(inputs).all().item():
            raise ValueError("inputs must contain only finite values")
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("model must have trainable parameters")
        if any(parameter.device != parameters[0].device for parameter in parameters):
            raise ValueError("trainable model parameters must be on the same device")
        if inputs.device != parameters[0].device:
            raise ValueError("inputs and model parameters must be on the same device")
        parameter_count = sum(parameter.numel() for parameter in parameters)
        sample_importances = []
        training_states = [(module, module.training) for module in model.modules()]
        model.eval()
        try:
            for index in range(inputs.shape[0]):
                with torch.enable_grad():
                    output = model(inputs[index : index + 1])
                    if not isinstance(output, torch.Tensor):
                        raise TypeError("model output must be a tensor")
                    if not torch.isfinite(output).all().item():
                        raise ValueError("model output must contain only finite values")
                    if output.requires_grad:
                        gradients = torch.autograd.grad(
                            output.abs().square().sum(),
                            parameters,
                            allow_unused=True,
                        )
                    else:
                        gradients = (None,) * len(parameters)
                flattened = torch.cat(
                    [
                        (torch.zeros_like(parameter) if gradient is None else gradient)
                        .detach()
                        .reshape(-1)
                        .abs()
                        for parameter, gradient in zip(parameters, gradients)
                    ]
                )
                if flattened.numel() != parameter_count or not torch.isfinite(flattened).all().item():
                    raise ValueError("model parameter sensitivities must be finite")
                sample_importances.append(flattened)
        finally:
            for module, training in training_states:
                module.training = training
        return torch.stack(sample_importances).mean(dim=0)

    def accumulate_importance(self, previous_importance, phase_importance, current_parameters):
        self._validate_vector(previous_importance, "previous_importance", nonnegative=True)
        self._validate_vector(phase_importance, "phase_importance", nonnegative=True)
        self._validate_vector(current_parameters, "current_parameters")
        vectors = (previous_importance, phase_importance, current_parameters)
        if any(value.shape != previous_importance.shape for value in vectors[1:]):
            raise ValueError("previous_importance, phase_importance, and current_parameters must have the same shape")
        if any(value.device != previous_importance.device for value in vectors[1:]):
            raise ValueError("importance and parameter vectors must be on the same device")
        return previous_importance + phase_importance, current_parameters.detach().clone()

    def forward(self, current_parameters, anchor_parameters, importance):
        self._validate_vector(current_parameters, "current_parameters", allow_batch=True)
        self._validate_vector(anchor_parameters, "anchor_parameters", allow_batch=True)
        self._validate_vector(importance, "importance", nonnegative=True, allow_batch=True)
        vectors = (current_parameters, anchor_parameters, importance)
        if any(value.shape != current_parameters.shape for value in vectors[1:]):
            raise ValueError("current_parameters, anchor_parameters, and importance must have the same shape")
        if any(value.device != current_parameters.device for value in vectors[1:]):
            raise ValueError("parameter and importance vectors must be on the same device")
        return self.lambda_ * (
            importance * (current_parameters - anchor_parameters).square()
        ).sum(dim=-1)
