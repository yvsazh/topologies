import math

import torch
from torch import nn


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, tradeoff):
        ctx.tradeoff = tradeoff
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.tradeoff * gradient, None


class CDAN(nn.Module):
    component_id = "cdan"

    def __init__(self, feature_dim, num_classes, tradeoff=1.0, randomized_dim=None):
        super().__init__()
        if type(feature_dim) is not int or feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer.")
        if type(num_classes) is not int or num_classes <= 0:
            raise ValueError("num_classes must be a positive integer.")
        if type(tradeoff) not in (int, float) or not math.isfinite(tradeoff) or tradeoff < 0:
            raise ValueError("tradeoff must be a finite nonnegative number.")
        if randomized_dim is not None and (type(randomized_dim) is not int or randomized_dim <= 0):
            raise ValueError("randomized_dim must be a positive integer when provided.")
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.tradeoff = float(tradeoff)
        self.randomized_dim = randomized_dim
        self.randomized = feature_dim * num_classes > 4096
        if self.randomized and randomized_dim is None:
            raise ValueError(
                "randomized_dim is required when feature_dim * num_classes exceeds 4096."
            )
        if not self.randomized and randomized_dim is not None:
            raise ValueError(
                "randomized_dim must be None when feature_dim * num_classes is at most 4096."
            )
        if self.randomized:
            self.register_buffer(
                "feature_projection", torch.randn(randomized_dim, feature_dim)
            )
            self.register_buffer(
                "class_projection", torch.randn(randomized_dim, num_classes)
            )

    def _validate_probabilities(self, probabilities):
        if not isinstance(probabilities, torch.Tensor) or probabilities.ndim != 2:
            raise ValueError("class probabilities must be a two-dimensional tensor.")
        if probabilities.shape[1] != self.num_classes:
            raise ValueError(f"class probabilities must have num_classes={self.num_classes}.")
        if not torch.is_floating_point(probabilities):
            raise ValueError("class probabilities must have a floating-point dtype.")
        if not torch.isfinite(probabilities).all():
            raise ValueError("class probabilities must contain only finite values.")
        if (probabilities < 0).any() or (probabilities > 1).any():
            raise ValueError("class probabilities must lie between 0 and 1.")
        totals = probabilities.sum(dim=1)
        if not torch.allclose(totals, torch.ones_like(totals), rtol=1e-4, atol=1e-4):
            raise ValueError("each class probability row must sum to 1.")

    def entropy_weights(self, probabilities):
        self._validate_probabilities(probabilities)
        tiny = torch.finfo(probabilities.dtype).tiny
        entropy = -(probabilities * probabilities.clamp_min(tiny).log()).sum(dim=1)
        reversed_entropy = _GradientReversal.apply(entropy, self.tradeoff)
        return 1.0 + torch.exp(-reversed_entropy)

    def forward(self, features, probabilities):
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise ValueError("features must be a two-dimensional tensor.")
        if features.shape[1] != self.feature_dim:
            raise ValueError(f"features must have feature_dim={self.feature_dim}.")
        self._validate_probabilities(probabilities)
        if features.shape[0] != probabilities.shape[0]:
            raise ValueError("features and class probabilities must have the same batch size.")
        if not torch.is_floating_point(features):
            raise ValueError("features must have a floating-point dtype.")
        if features.device != probabilities.device or features.dtype != probabilities.dtype:
            raise ValueError("features and class probabilities must share device and dtype.")
        reversed_features = _GradientReversal.apply(features, self.tradeoff)
        reversed_probabilities = _GradientReversal.apply(probabilities, self.tradeoff)
        if self.randomized:
            feature_projection = self.feature_projection.to(
                device=features.device, dtype=features.dtype
            )
            class_projection = self.class_projection.to(
                device=probabilities.device, dtype=probabilities.dtype
            )
            feature_map = reversed_features @ feature_projection.transpose(0, 1)
            class_map = reversed_probabilities @ class_projection.transpose(0, 1)
            return feature_map * class_map / math.sqrt(self.randomized_dim)
        return (
            reversed_features.unsqueeze(2) * reversed_probabilities.unsqueeze(1)
        ).flatten(start_dim=1)
