import math
import numbers

import torch
import torch.nn.functional as F
from torch import nn


class MixMatch(nn.Module):
    component_id = "mixmatch"

    def __init__(
        self,
        temperature=0.5,
        num_augmentations=2,
        alpha=0.75,
        lambda_u_max=100.0,
        rampup_steps=16000,
    ):
        super().__init__()
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, numbers.Real)
            or not math.isfinite(temperature)
            or temperature <= 0
        ):
            raise ValueError("temperature must be a finite real number greater than zero")
        if (
            isinstance(num_augmentations, bool)
            or not isinstance(num_augmentations, numbers.Integral)
            or num_augmentations < 1
        ):
            raise ValueError("num_augmentations must be a positive integer")
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, numbers.Real)
            or not math.isfinite(alpha)
            or alpha <= 0
        ):
            raise ValueError("alpha must be a finite real number greater than zero")
        if (
            isinstance(lambda_u_max, bool)
            or not isinstance(lambda_u_max, numbers.Real)
            or not math.isfinite(lambda_u_max)
            or lambda_u_max < 0
        ):
            raise ValueError("lambda_u_max must be a finite nonnegative real number")
        if (
            isinstance(rampup_steps, bool)
            or not isinstance(rampup_steps, numbers.Integral)
            or rampup_steps < 1
        ):
            raise ValueError("rampup_steps must be a positive integer")
        self.temperature = float(temperature)
        self.num_augmentations = int(num_augmentations)
        self.alpha = float(alpha)
        self.lambda_u_max = float(lambda_u_max)
        self.rampup_steps = int(rampup_steps)

    def guess_labels(self, unlabeled_probabilities):
        if not isinstance(unlabeled_probabilities, torch.Tensor):
            raise TypeError("unlabeled_probabilities must be a tensor")
        if unlabeled_probabilities.ndim != 3:
            raise ValueError("unlabeled_probabilities must have shape [B, K, C]")
        if unlabeled_probabilities.shape[1] != self.num_augmentations:
            raise ValueError(
                "unlabeled_probabilities K dimension must equal num_augmentations"
            )
        if not unlabeled_probabilities.is_floating_point():
            raise ValueError("unlabeled_probabilities must have a floating-point dtype")
        if unlabeled_probabilities.shape[0] < 1 or unlabeled_probabilities.shape[2] < 1:
            raise ValueError("unlabeled_probabilities must have nonempty batch and class dimensions")
        if not torch.isfinite(unlabeled_probabilities).all():
            raise ValueError("unlabeled_probabilities must contain only finite values")
        if (unlabeled_probabilities < 0).any():
            raise ValueError("unlabeled_probabilities must be nonnegative")
        if (unlabeled_probabilities.sum(dim=-1) <= 0).any():
            raise ValueError("each unlabeled probability distribution must have positive mass")
        averaged = unlabeled_probabilities.mean(dim=1)
        sharpened = averaged.pow(1.0 / self.temperature)
        guessed = sharpened / sharpened.sum(dim=-1, keepdim=True)
        return guessed.detach()

    def forward(
        self,
        labeled_inputs,
        labeled_targets,
        unlabeled_views,
        unlabeled_probabilities,
        permutation=None,
        lambdas=None,
    ):
        mixed_labeled_inputs, mixed_labeled_targets, mixed_unlabeled_inputs, mixed_unlabeled_targets = self.mix_batch(
            labeled_inputs,
            labeled_targets,
            unlabeled_views,
            unlabeled_probabilities,
            permutation=permutation,
            lambdas=lambdas,
        )
        batch_size = labeled_inputs.shape[0]
        mixed_unlabeled_inputs = mixed_unlabeled_inputs.reshape(
            batch_size, self.num_augmentations, *labeled_inputs.shape[1:]
        )
        return torch.cat(
            (
                mixed_labeled_inputs.flatten(start_dim=1),
                mixed_labeled_targets,
                mixed_unlabeled_inputs.flatten(start_dim=1),
                mixed_unlabeled_targets.reshape(batch_size, -1),
            ),
            dim=-1,
        )

    def mix_batch(
        self,
        labeled_inputs,
        labeled_targets,
        unlabeled_views,
        unlabeled_probabilities,
        permutation=None,
        lambdas=None,
    ):
        values = {
            "labeled_inputs": labeled_inputs,
            "labeled_targets": labeled_targets,
            "unlabeled_views": unlabeled_views,
            "unlabeled_probabilities": unlabeled_probabilities,
        }
        for name, value in values.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a tensor")
        if labeled_inputs.ndim < 2:
            raise ValueError("labeled_inputs must have shape [B, ...]")
        if unlabeled_views.ndim < 3:
            raise ValueError("unlabeled_views must have shape [B, K, ...]")
        if labeled_targets.ndim != 2:
            raise ValueError("labeled_targets must have shape [B, C]")
        if unlabeled_probabilities.ndim != 3:
            raise ValueError("unlabeled_probabilities must have shape [B, K, C]")
        batch_size = labeled_inputs.shape[0]
        if batch_size < 1 or labeled_targets.shape[0] != batch_size:
            raise ValueError("labeled_inputs and labeled_targets must have the same nonempty batch")
        if unlabeled_views.shape[0] != batch_size:
            raise ValueError("unlabeled_views batch dimension must match labeled_inputs")
        if unlabeled_views.shape[1] != self.num_augmentations:
            raise ValueError("unlabeled_views K dimension must equal num_augmentations")
        if unlabeled_probabilities.shape[:2] != unlabeled_views.shape[:2]:
            raise ValueError("unlabeled_probabilities B and K dimensions must match unlabeled_views")
        if unlabeled_views.shape[2:] != labeled_inputs.shape[1:]:
            raise ValueError("labeled_inputs and unlabeled_views must have matching feature shapes")
        num_classes = labeled_targets.shape[1]
        if num_classes < 1 or unlabeled_probabilities.shape[2] != num_classes:
            raise ValueError("labeled_targets and unlabeled_probabilities must have matching classes")
        if not labeled_inputs.is_floating_point() or not unlabeled_views.is_floating_point():
            raise ValueError("labeled_inputs and unlabeled_views must have floating-point dtypes")
        if labeled_inputs.dtype != unlabeled_views.dtype:
            raise ValueError("labeled_inputs and unlabeled_views must have matching dtypes")
        if not labeled_targets.is_floating_point():
            raise ValueError("labeled_targets must have a floating-point dtype")
        tensors = (labeled_inputs, labeled_targets, unlabeled_views, unlabeled_probabilities)
        if any(value.device != labeled_inputs.device for value in tensors[1:]):
            raise ValueError("all inputs, targets, and probabilities must be on the same device")
        if not torch.isfinite(labeled_targets).all():
            raise ValueError("labeled_targets must contain only finite values")
        if (labeled_targets < 0).any():
            raise ValueError("labeled_targets must be nonnegative")

        guessed = self.guess_labels(unlabeled_probabilities)
        guessed = guessed.to(dtype=labeled_targets.dtype)
        total = batch_size * (1 + self.num_augmentations)
        source_inputs = torch.cat(
            (labeled_inputs, unlabeled_views.reshape(batch_size * self.num_augmentations, *labeled_inputs.shape[1:])),
            dim=0,
        )
        source_targets = torch.cat(
            (
                labeled_targets,
                guessed.repeat_interleave(self.num_augmentations, dim=0),
            ),
            dim=0,
        )
        if permutation is None:
            permutation = torch.randperm(total, device=labeled_inputs.device)
        else:
            if not isinstance(permutation, torch.Tensor):
                permutation = torch.as_tensor(permutation, device=labeled_inputs.device)
            if permutation.device != labeled_inputs.device:
                raise ValueError("permutation must be on the same device as the inputs")
            if permutation.dtype != torch.long or permutation.shape != (total,):
                raise ValueError("permutation must be a long tensor with one index per source")
            if not torch.equal(torch.sort(permutation).values, torch.arange(total, device=permutation.device)):
                raise ValueError("permutation must contain each source index exactly once")
        shuffled_inputs = source_inputs[permutation]
        shuffled_targets = source_targets[permutation]
        if lambdas is None:
            lambdas = torch.distributions.Beta(self.alpha, self.alpha).sample((total,))
        else:
            lambdas = torch.as_tensor(lambdas, device=labeled_inputs.device)
        lambdas = lambdas.to(device=labeled_inputs.device, dtype=labeled_inputs.dtype)
        if lambdas.shape != (total,):
            raise ValueError("lambdas must have one coefficient per source")
        if not torch.isfinite(lambdas).all() or (lambdas < 0).any() or (lambdas > 1).any():
            raise ValueError("lambdas must contain finite values in [0, 1]")
        clipped = torch.maximum(lambdas, 1.0 - lambdas)
        input_shape = (batch_size,) + (1,) * (labeled_inputs.ndim - 1)
        target_shape = (batch_size, 1)
        labeled_lambda = clipped[:batch_size].reshape(input_shape)
        labeled_target_lambda = clipped[:batch_size].reshape(target_shape)
        mixed_labeled_inputs = (
            labeled_lambda * labeled_inputs
            + (1.0 - labeled_lambda) * shuffled_inputs[:batch_size]
        )
        mixed_labeled_targets = (
            labeled_target_lambda * labeled_targets
            + (1.0 - labeled_target_lambda) * shuffled_targets[:batch_size]
        )
        unlabeled_count = batch_size * self.num_augmentations
        flat_unlabeled = unlabeled_views.reshape(unlabeled_count, *labeled_inputs.shape[1:])
        flat_guesses = guessed.repeat_interleave(self.num_augmentations, dim=0)
        unlabeled_lambda = clipped[batch_size:].reshape(
            (unlabeled_count,) + (1,) * (flat_unlabeled.ndim - 1)
        )
        unlabeled_target_lambda = clipped[batch_size:].reshape(unlabeled_count, 1)
        mixed_unlabeled_inputs = (
            unlabeled_lambda * flat_unlabeled
            + (1.0 - unlabeled_lambda) * shuffled_inputs[batch_size:]
        )
        mixed_unlabeled_targets = (
            unlabeled_target_lambda * flat_guesses
            + (1.0 - unlabeled_target_lambda) * shuffled_targets[batch_size:]
        )
        return (
            mixed_labeled_inputs,
            mixed_labeled_targets,
            mixed_unlabeled_inputs,
            mixed_unlabeled_targets,
        )

    def unsupervised_weight(self, step):
        if isinstance(step, bool) or not isinstance(step, numbers.Integral) or step < 0:
            raise ValueError("step must be a nonnegative integer")
        return self.lambda_u_max * min(int(step) / self.rampup_steps, 1.0)

    def compute_losses(
        self,
        labeled_logits,
        labeled_targets,
        unlabeled_logits,
        guessed_targets,
        step,
    ):
        pairs = (
            ("labeled_logits", labeled_logits),
            ("labeled_targets", labeled_targets),
            ("unlabeled_logits", unlabeled_logits),
            ("guessed_targets", guessed_targets),
        )
        for name, value in pairs:
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a tensor")
            if value.ndim != 2:
                raise ValueError(f"{name} must have shape [N, C]")
        if labeled_logits.shape != labeled_targets.shape:
            raise ValueError("labeled_logits and labeled_targets must have matching shapes")
        if unlabeled_logits.shape != guessed_targets.shape:
            raise ValueError("unlabeled_logits and guessed_targets must have matching shapes")
        if labeled_logits.shape[0] < 1 or unlabeled_logits.shape[0] < 1:
            raise ValueError("labeled and unlabeled logits must have nonempty batches")
        if labeled_logits.shape[1] < 1 or unlabeled_logits.shape[1] != labeled_logits.shape[1]:
            raise ValueError("labeled and unlabeled logits must have matching nonempty classes")
        if any(not value.is_floating_point() for _, value in pairs):
            raise ValueError("logits and targets must have floating-point dtypes")
        if any(value.device != labeled_logits.device for _, value in pairs[1:]):
            raise ValueError("logits and targets must be on the same device")
        if not torch.isfinite(labeled_targets).all() or (labeled_targets < 0).any():
            raise ValueError("labeled_targets must be finite and nonnegative")
        if not torch.isfinite(guessed_targets).all() or (guessed_targets < 0).any():
            raise ValueError("guessed_targets must be finite and nonnegative")
        weight = self.unsupervised_weight(step)
        supervised = -(
            labeled_targets.to(dtype=labeled_logits.dtype)
            * F.log_softmax(labeled_logits, dim=-1)
        ).sum(dim=-1).mean()
        probabilities = F.softmax(unlabeled_logits, dim=-1)
        unsupervised = (
            probabilities - guessed_targets.to(dtype=unlabeled_logits.dtype)
        ).square().mean()
        total = supervised + weight * unsupervised
        return {
            "supervised": supervised,
            "unsupervised": unsupervised,
            "weight": weight,
            "total": total,
        }
