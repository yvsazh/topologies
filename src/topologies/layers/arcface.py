import math

import torch
from torch import nn


class ArcFace(nn.Module):
    component_id = "arcface"

    def __init__(self, num_classes, embedding_dim=512, scale=64.0, margin=0.5):
        super().__init__()
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")
        if isinstance(embedding_dim, bool) or not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive integer")
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be a finite positive number")
        if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(margin) or margin < 0:
            raise ValueError("margin must be a finite nonnegative number")
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.scale = float(scale)
        self.margin = float(margin)
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels):
        if features.ndim != 2:
            raise ValueError("features must have shape [batch, embedding_dim]")
        if features.shape[1] != self.embedding_dim:
            raise ValueError("features embedding dimension does not match embedding_dim")
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
            raise ValueError("labels must have shape [batch] matching the feature batch size")
        if labels.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("labels must have an integer dtype")
        if not features.is_floating_point():
            raise ValueError("features must have a floating point dtype")
        if features.dtype != self.weight.dtype:
            raise ValueError("features and class vectors must have the same dtype")
        if features.device != self.weight.device or labels.device != features.device:
            raise ValueError("features, labels, and class vectors must be on the same device")
        if labels.numel() and (labels.min() < 0 or labels.max() >= self.num_classes):
            raise ValueError("labels must contain class indices in range [0, num_classes)")
        normalized_features = torch.nn.functional.normalize(features, p=2, dim=1)
        normalized_weight = torch.nn.functional.normalize(self.weight, p=2, dim=1)
        cosine = (normalized_features @ normalized_weight.transpose(0, 1)).clamp(-1.0, 1.0)
        indices = labels.to(dtype=torch.long).unsqueeze(1)
        target_cosine = cosine.gather(1, indices)
        epsilon = torch.finfo(target_cosine.dtype).eps
        stable_target_cosine = target_cosine.clamp(-1.0 + epsilon, 1.0 - epsilon)
        target_logits = torch.cos(torch.acos(stable_target_cosine) + self.margin)
        target_logits = torch.where(target_cosine == 1.0, math.cos(self.margin), target_logits)
        target_logits = torch.where(target_cosine == -1.0, math.cos(math.pi + self.margin), target_logits)
        return cosine.scatter(1, indices, target_logits) * self.scale
