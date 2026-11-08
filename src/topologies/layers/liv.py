import math
import numbers

import torch
from torch import nn


class LIV(nn.Module):
    component_id = "liv"

    def __init__(self, gamma=0.98, language_vip_weight=0.0, reduction="mean", *args, **kwargs):
        if args or kwargs:
            raise ValueError("LIV accepts only gamma, language_vip_weight, and reduction")
        super().__init__()
        for name, value in (("gamma", gamma), ("language_vip_weight", language_vip_weight)):
            if not self._is_real_number(value):
                raise ValueError(f"{name} must be a finite real number")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite real number")
            if name == "gamma" and not 0.0 <= value < 1.0:
                raise ValueError("gamma must be in [0, 1)")
            if name == "language_vip_weight" and value < 0.0:
                raise ValueError("language_vip_weight must be non-negative")
            setattr(self, name, value)
        if not isinstance(reduction, str) or reduction not in {"mean", "none"}:
            raise ValueError("reduction must be 'mean' or 'none'")
        self.reduction = reduction

    @staticmethod
    def _is_real_number(value):
        return issubclass(type(value), numbers.Real) and not issubclass(type(value), bool)

    @staticmethod
    def _validate_embeddings(*values):
        if any(not isinstance(value, torch.Tensor) for value in values):
            raise ValueError("all embeddings must be torch.Tensor instances")
        if any(value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0 for value in values):
            raise ValueError("embeddings must have non-empty shape [B, D]")
        if any(not value.is_floating_point() for value in values):
            raise ValueError("embeddings must have floating-point dtypes")
        reference = values[0]
        if any(value.shape != reference.shape for value in values[1:]):
            raise ValueError("embeddings must have matching shape [B, D]")
        if any(value.dtype != reference.dtype or value.device != reference.device for value in values[1:]):
            raise ValueError("embeddings must share dtype and device")
        return reference.shape[0]

    def scaled_similarity(self, values, goals):
        self._validate_embeddings(values, goals)
        return torch.nn.functional.cosine_similarity(values, goals, dim=-1) / (1.0 - self.gamma)

    def _vip_loss_per_example(self, initial, current, next_value, goal):
        batch_size = self._validate_embeddings(initial, current, next_value, goal)
        initial_similarity = self.scaled_similarity(initial, goal)
        current_similarity = self.scaled_similarity(current, goal)
        next_similarity = self.scaled_similarity(next_value, goal)
        temporal_terms = current_similarity + 1.0 - self.gamma * next_similarity
        temporal_normalizer = torch.logsumexp(temporal_terms, dim=0) - math.log(batch_size)
        return -(1.0 - self.gamma) * initial_similarity + temporal_normalizer

    def vip_loss(self, initial, current, next_value, goal):
        return self._vip_loss_per_example(initial, current, next_value, goal).mean()

    def _info_nce_per_example(self, image_goal, text_goal):
        batch_size = self._validate_embeddings(image_goal, text_goal)
        pairwise_scores = torch.nn.functional.normalize(image_goal, dim=-1) @ (
            torch.nn.functional.normalize(text_goal, dim=-1).transpose(0, 1)
        )
        log_mean_exp = torch.logsumexp(pairwise_scores, dim=0) - math.log(batch_size)
        positive_scores = pairwise_scores.diagonal()
        return (1.0 - self.gamma) * (log_mean_exp - positive_scores)

    def info_nce_loss(self, image_goal, text_goal):
        return self._info_nce_per_example(image_goal, text_goal).mean()

    def reward(self, current, next_value, goal):
        self._validate_embeddings(current, next_value, goal)
        return self.scaled_similarity(next_value, goal) - self.scaled_similarity(current, goal)

    def forward(self, initial, current, next_value, image_goal, text_goal):
        self._validate_embeddings(initial, current, next_value, image_goal, text_goal)
        per_example_loss = self._vip_loss_per_example(initial, current, next_value, image_goal)
        per_example_loss = per_example_loss + self._info_nce_per_example(image_goal, text_goal)
        if self.language_vip_weight:
            per_example_loss = per_example_loss + self.language_vip_weight * self._vip_loss_per_example(
                initial, current, next_value, text_goal
            )
        return per_example_loss if self.reduction == "none" else per_example_loss.mean()
