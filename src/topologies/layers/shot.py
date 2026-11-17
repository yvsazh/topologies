import math
import numbers

import torch
from torch import nn
from torch.nn import functional as F


def _is_real_number(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


class SHOT(nn.Module):
    component_id = "shot"
    PRESETS = {
        "tiny": {"feature_dim": 2, "num_classes": 2},
        "small": {"feature_dim": 8, "num_classes": 4},
    }

    def __init__(self, feature_dim=2, num_classes=2, beta=0.3, encoder=None, classifier=None):
        super().__init__()
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim < 1:
            raise ValueError("feature_dim must be a positive integer")
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError("num_classes must be an integer of at least two")
        if not _is_real_number(beta):
            raise ValueError("beta must be a finite nonnegative number")
        if not math.isfinite(beta) or beta < 0:
            raise ValueError("beta must be a finite nonnegative number")
        if encoder is not None and not isinstance(encoder, nn.Module):
            raise ValueError("encoder must be a torch.nn.Module")
        if classifier is not None and not isinstance(classifier, nn.Module):
            raise ValueError("classifier must be a torch.nn.Module")
        self.encoder = encoder if encoder is not None else nn.Linear(feature_dim, feature_dim)
        self.classifier = classifier if classifier is not None else nn.Linear(feature_dim, num_classes)
        self.beta = float(beta)
        self.classifier.requires_grad_(False)
        self.classifier.eval()

    def train(self, mode=True):
        super().train(mode)
        self.classifier.eval()
        return self

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim < 2 or x.shape[0] < 1:
            raise ValueError("x must contain a nonempty batch dimension")
        if not x.is_floating_point():
            raise ValueError("x must have a floating-point dtype")
        features = self.encoder(x)
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise ValueError("encoder must return a two-dimensional tensor shaped [B, D]")
        if features.shape[0] != x.shape[0] or features.shape[1] < 1:
            raise ValueError("encoder output must preserve the batch dimension and have features")
        logits = self.classifier(features)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
            raise ValueError("classifier must return a two-dimensional tensor shaped [B, K]")
        if logits.shape[0] != x.shape[0] or logits.shape[1] < 2:
            raise ValueError("classifier output must preserve the batch dimension and have at least two classes")
        if not logits.is_floating_point():
            raise ValueError("classifier output must have a floating-point dtype")
        return logits

    @staticmethod
    def _validate_probabilities(probabilities):
        if not isinstance(probabilities, torch.Tensor):
            raise TypeError("probabilities must be a torch.Tensor")
        if probabilities.ndim != 2 or probabilities.shape[0] < 1 or probabilities.shape[1] < 2:
            raise ValueError("probabilities must have shape [N, K] with N positive and K at least two")
        if not probabilities.is_floating_point():
            raise ValueError("probabilities must have a floating-point dtype")
        if not torch.isfinite(probabilities).all().item() or torch.any(probabilities < 0).item():
            raise ValueError("probabilities must be finite and nonnegative")
        totals = probabilities.sum(dim=1)
        if not torch.allclose(totals, torch.ones_like(totals), rtol=1e-4, atol=1e-5):
            raise ValueError("each probability row must sum to one")

    @staticmethod
    def information_maximization(probabilities):
        SHOT._validate_probabilities(probabilities)
        tiny = torch.finfo(probabilities.dtype).tiny
        sample_entropy = -(probabilities * probabilities.clamp_min(tiny).log()).sum(dim=1).mean()
        mean_probabilities = probabilities.mean(dim=0)
        diversity = (mean_probabilities * mean_probabilities.clamp_min(tiny).log()).sum()
        return sample_entropy + diversity

    @staticmethod
    def _nearest_centroid_labels(features, centroids):
        normalized_features = F.normalize(features, dim=1, eps=1e-12)
        normalized_centroids = F.normalize(centroids, dim=1, eps=1e-12)
        similarities = normalized_features @ normalized_centroids.transpose(0, 1)
        return similarities.argmax(dim=1)

    @staticmethod
    def pseudo_label_details(features, probabilities):
        SHOT._validate_probabilities(probabilities)
        if not isinstance(features, torch.Tensor):
            raise TypeError("features must be a torch.Tensor")
        if features.ndim != 2 or features.shape[0] < 1 or features.shape[1] < 1:
            raise ValueError("features must have shape [N, D] with positive N and D")
        if not features.is_floating_point():
            raise ValueError("features must have a floating-point dtype")
        if features.shape[0] != probabilities.shape[0]:
            raise ValueError("features and probabilities must have the same number of examples")
        if features.device != probabilities.device:
            raise ValueError("features and probabilities must be on the same device")
        if not torch.isfinite(features).all().item():
            raise ValueError("features must be finite")
        masses = probabilities.sum(dim=0)
        if torch.any(masses <= 0).item():
            raise ValueError("every class must have positive weighted centroid mass")
        initial_centroids = probabilities.transpose(0, 1) @ features / masses.unsqueeze(1)
        initial_labels = SHOT._nearest_centroid_labels(features, initial_centroids)
        counts = torch.bincount(initial_labels, minlength=probabilities.shape[1])
        if torch.any(counts == 0).item():
            raise ValueError("pseudo-label assignment produced an empty refined centroid")
        refined_centroids = torch.stack(
            [features[initial_labels == class_index].mean(dim=0)
             for class_index in range(probabilities.shape[1])]
        )
        labels = SHOT._nearest_centroid_labels(features, refined_centroids)
        return initial_centroids, labels, refined_centroids

    @staticmethod
    def pseudo_labels(features, probabilities):
        return SHOT.pseudo_label_details(features, probabilities)[1]

    def loss(self, logits, pseudo_labels=None):
        if not isinstance(logits, torch.Tensor):
            raise TypeError("logits must be a torch.Tensor")
        if logits.ndim != 2 or logits.shape[0] < 1 or logits.shape[1] < 2:
            raise ValueError("logits must have shape [B, K] with B positive and K at least two")
        if not logits.is_floating_point() or not torch.isfinite(logits).all().item():
            raise ValueError("logits must have finite floating-point values")
        probabilities = torch.softmax(logits, dim=1)
        information_loss = self.information_maximization(probabilities)
        if pseudo_labels is None:
            if self.beta != 0:
                raise ValueError("pseudo_labels are required when beta is positive")
            return information_loss
        if not isinstance(pseudo_labels, torch.Tensor):
            raise TypeError("pseudo_labels must be a torch.Tensor")
        if pseudo_labels.ndim != 1 or pseudo_labels.shape[0] != logits.shape[0]:
            raise ValueError("pseudo_labels must have shape [B] matching logits")
        if pseudo_labels.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("pseudo_labels must contain integer class indices")
        if pseudo_labels.device != logits.device:
            raise ValueError("pseudo_labels and logits must be on the same device")
        if torch.any(pseudo_labels < 0).item() or torch.any(pseudo_labels >= logits.shape[1]).item():
            raise ValueError("pseudo_labels must be valid class indices for logits")
        return information_loss + self.beta * F.cross_entropy(logits, pseudo_labels.long())
