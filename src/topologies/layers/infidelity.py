import torch
from torch import nn


class Infidelity(nn.Module):
    component_id = "infidelity"

    def __init__(self):
        super().__init__()

    def forward(self, inputs, explanation, perturbations, predictor):
        if not all(isinstance(value, torch.Tensor) for value in (inputs, explanation, perturbations)):
            raise ValueError("inputs, explanation, and perturbations must be torch tensors")
        if inputs.ndim < 1:
            raise ValueError("inputs must have at least one feature dimension")
        if inputs.numel() == 0 or (inputs.ndim > 1 and inputs.shape[0] == 0):
            raise ValueError("inputs must contain at least one example and feature")
        if tuple(explanation.shape) != tuple(inputs.shape):
            raise ValueError("explanation must have the same shape as inputs")
        if perturbations.ndim < 2:
            raise ValueError("perturbations must include sample and feature dimensions")
        if not all(value.is_floating_point() for value in (inputs, explanation, perturbations)):
            raise ValueError("inputs, explanation, and perturbations must be floating-point tensors")
        if any(value.device != inputs.device for value in (explanation, perturbations)):
            raise ValueError("inputs, explanation, and perturbations must be on the same device")
        if any(value.dtype != inputs.dtype for value in (explanation, perturbations)):
            raise ValueError("inputs, explanation, and perturbations must have the same dtype")

        batched_weights = (
            torch.is_tensor(predictor)
            and inputs.ndim >= 2
            and predictor.ndim == inputs.ndim
            and tuple(predictor.shape) == tuple(inputs.shape)
        )
        if batched_weights:
            if perturbations.ndim == inputs.ndim and tuple(perturbations.shape) == tuple(inputs.shape):
                perturbations = perturbations.unsqueeze(1)
            if (
                perturbations.ndim != inputs.ndim + 1
                or perturbations.shape[0] != inputs.shape[0]
                or tuple(perturbations.shape[2:]) != tuple(inputs.shape[1:])
            ):
                raise ValueError("batched perturbations must have shape [batch, samples, *features]")
            if perturbations.shape[1] < 1:
                raise ValueError("perturbations must contain at least one sample")
            if not predictor.is_floating_point():
                raise ValueError("linear predictor weights must be floating-point")
            if predictor.device != inputs.device:
                raise ValueError("linear predictor weights must be on the same device as inputs")
            if predictor.dtype != inputs.dtype:
                raise ValueError("linear predictor weights must have the same dtype as inputs")
            sample_count = perturbations.shape[1]
            flat_perturbations = perturbations.flatten(start_dim=2)
            flat_weights = predictor.flatten(start_dim=1)
            output_change = torch.einsum("bkd,bd->bk", flat_perturbations, flat_weights)
            explanation_change = (perturbations * explanation.unsqueeze(1)).flatten(start_dim=2).sum(dim=2)
            return (explanation_change - output_change).square().mean(dim=1)

        if perturbations.ndim != inputs.ndim + 1 or tuple(perturbations.shape[1:]) != tuple(inputs.shape):
            raise ValueError("perturbations must have a leading sample dimension and match inputs")
        if perturbations.shape[0] < 1:
            raise ValueError("perturbations must contain at least one sample")
        sample_count = perturbations.shape[0]
        feature_count = inputs.numel()
        flat_perturbations = perturbations.reshape(sample_count, feature_count)

        if torch.is_tensor(predictor):
            if predictor.ndim != 1 or predictor.numel() != feature_count:
                raise ValueError("linear predictor weights must be a vector matching the input features")
            if not predictor.is_floating_point():
                raise ValueError("linear predictor weights must be floating-point")
            if predictor.device != inputs.device:
                raise ValueError("linear predictor weights must be on the same device as inputs")
            if predictor.dtype != inputs.dtype:
                raise ValueError("linear predictor weights must have the same dtype as inputs")
            output_change = flat_perturbations @ predictor
        elif callable(predictor):
            original_output = predictor(inputs)
            perturbed_output = predictor(inputs.unsqueeze(0) - perturbations)
            if not torch.is_tensor(original_output) or not torch.is_tensor(perturbed_output):
                raise ValueError("predictor must return torch tensors")
            if not original_output.is_floating_point() or not perturbed_output.is_floating_point():
                raise ValueError("predictor outputs must be floating-point scalars")
            if original_output.device != inputs.device or perturbed_output.device != inputs.device:
                raise ValueError("predictor outputs must be on the same device as inputs")
            if original_output.numel() != 1:
                raise ValueError("predictor must return one scalar for inputs")
            if perturbed_output.numel() != sample_count or (
                perturbed_output.ndim > 0 and perturbed_output.shape[0] != sample_count
            ):
                raise ValueError("predictor must return one scalar per perturbation")
            output_change = original_output.reshape(()) - perturbed_output.reshape(sample_count)
        else:
            raise ValueError("predictor must be callable or a linear weight tensor")

        explanation_change = (perturbations * explanation.unsqueeze(0)).reshape(sample_count, -1).sum(dim=1)
        return (explanation_change - output_change).square().mean()
