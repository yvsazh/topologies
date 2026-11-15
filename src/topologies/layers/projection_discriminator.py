import torch
from torch import nn


class ProjectionDiscriminator(nn.Module):
    component_id = "projection-discriminator"
    PRESETS = {
        "tiny": {"feature_dim": 4, "num_classes": 3, "score_bias": True},
        "small": {"feature_dim": 16, "num_classes": 10, "score_bias": True},
    }

    def __init__(self, feature_dim, num_classes, score_bias=True):
        super().__init__()
        if not isinstance(feature_dim, int) or isinstance(feature_dim, bool) or feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer")
        if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")
        if score_bias is not True and score_bias is not False:
            raise ValueError("score_bias must be a bool")
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.score_bias = score_bias
        self.class_embedding = nn.Embedding(num_classes, feature_dim)
        self.unconditional_score = nn.Linear(feature_dim, 1, bias=score_bias)

    def forward(self, features, labels):
        if not isinstance(features, torch.Tensor):
            raise TypeError("features must be a tensor")
        if features.ndim != 2:
            raise ValueError("features must have shape [B, feature_dim]")
        if not features.is_floating_point():
            raise ValueError("features must have a floating-point dtype")
        if features.shape[1] != self.feature_dim:
            raise ValueError("feature width must match feature_dim")
        if not isinstance(labels, torch.Tensor):
            raise TypeError("labels must be a tensor")
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
            raise ValueError("labels must have shape [B]")
        if labels.dtype != torch.long:
            raise ValueError("labels must have dtype torch.long")
        if labels.device != features.device:
            raise ValueError("features and labels must be on the same device")
        if labels.numel() and bool(((labels < 0) | (labels >= self.num_classes)).any()):
            raise ValueError("labels must be in [0, num_classes)")
        projected = (self.class_embedding(labels) * features).sum(dim=1, keepdim=True)
        return projected + self.unconditional_score(features)
