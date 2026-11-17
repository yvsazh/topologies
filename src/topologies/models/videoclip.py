import math

import torch
from torch import nn
from torch.nn import functional as F


class VideoCLIP(nn.Module):
    architecture_id = "videoclip"
    PRESETS = {
        "tiny": {
            "video_feature_dim": 512,
            "hidden_size": 32,
            "video_layers": 1,
            "text_layers": 1,
            "attention_heads": 4,
            "ff_dim": 128,
            "temperature": 1.0,
        },
        "small": {
            "video_feature_dim": 512,
            "hidden_size": 128,
            "video_layers": 2,
            "text_layers": 3,
            "attention_heads": 4,
            "ff_dim": 512,
            "temperature": 1.0,
        },
    }

    def __init__(
        self,
        video_feature_dim=512,
        hidden_size=32,
        video_layers=1,
        text_layers=1,
        attention_heads=4,
        ff_dim=128,
        temperature=1.0,
    ):
        super().__init__()
        dimensions = {
            "video_feature_dim": video_feature_dim,
            "hidden_size": hidden_size,
            "video_layers": video_layers,
            "text_layers": text_layers,
            "attention_heads": attention_heads,
            "ff_dim": ff_dim,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if hidden_size % attention_heads != 0:
            raise ValueError("hidden_size must be divisible by attention_heads")
        if not VideoCLIP._is_valid_temperature(temperature):
            raise ValueError("temperature must be a finite positive number")
        self.video_feature_dim = video_feature_dim
        self.hidden_size = hidden_size
        self.temperature = float(temperature)
        self.video_projection = nn.Sequential(
            nn.Linear(video_feature_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.video_transformer = self._make_transformer(hidden_size, attention_heads, ff_dim, video_layers)
        self.text_transformer = self._make_transformer(hidden_size, attention_heads, ff_dim, text_layers)

    @staticmethod
    def _make_transformer(hidden_size, attention_heads, ff_dim, layers):
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=attention_heads,
            dim_feedforward=ff_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        return nn.TransformerEncoder(layer, num_layers=layers)

    @staticmethod
    def _is_tensor(value):
        return isinstance(value, torch.Tensor)

    @staticmethod
    def _is_valid_temperature(value):
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
            and value > 0
        )

    @staticmethod
    def _validate_embeddings(video_embeddings, text_embeddings, temperature):
        if not VideoCLIP._is_tensor(video_embeddings) or not VideoCLIP._is_tensor(text_embeddings):
            raise ValueError("video_embeddings and text_embeddings must be tensors")
        if video_embeddings.ndim != 2 or text_embeddings.ndim != 2:
            raise ValueError("embeddings must have shape (batch, hidden_size)")
        if video_embeddings.shape[0] <= 0 or video_embeddings.shape[1] <= 0:
            raise ValueError("embeddings must have nonempty batch and feature dimensions")
        if video_embeddings.shape != text_embeddings.shape:
            raise ValueError("video_embeddings and text_embeddings must have matching shapes")
        if not VideoCLIP._is_valid_temperature(temperature):
            raise ValueError("temperature must be a finite positive number")

    @staticmethod
    def similarity(video_embeddings, text_embeddings, temperature=1.0):
        VideoCLIP._validate_embeddings(video_embeddings, text_embeddings, temperature)
        return video_embeddings @ text_embeddings.transpose(0, 1) / temperature

    @staticmethod
    def contrastive_loss(video_embeddings, text_embeddings, temperature=1.0):
        logits = VideoCLIP.similarity(video_embeddings, text_embeddings, temperature)
        targets = torch.arange(logits.shape[0], device=logits.device)
        return F.cross_entropy(logits, targets, reduction="sum") + F.cross_entropy(
            logits.transpose(0, 1), targets, reduction="sum"
        )

    def forward(self, video_features, text_embeddings):
        if not VideoCLIP._is_tensor(video_features) or not VideoCLIP._is_tensor(text_embeddings):
            raise ValueError("video_features and text_embeddings must be tensors")
        if video_features.ndim != 3 or text_embeddings.ndim != 3:
            raise ValueError("video_features and text_embeddings must have shape (batch, sequence, features)")
        if video_features.shape[0] <= 0 or video_features.shape[1] <= 0 or text_embeddings.shape[1] <= 0:
            raise ValueError("video and text inputs must have nonempty batch and sequence dimensions")
        if video_features.shape[0] != text_embeddings.shape[0]:
            raise ValueError("video_features and text_embeddings must have matching batch sizes")
        if video_features.shape[2] != self.video_feature_dim:
            raise ValueError("video_features have the wrong feature width")
        if text_embeddings.shape[2] != self.hidden_size:
            raise ValueError("text_embeddings have the wrong feature width")
        if not video_features.is_floating_point() or not text_embeddings.is_floating_point():
            raise ValueError("video_features and text_embeddings must have floating point values")
        video_hidden = self.video_transformer(self.video_projection(video_features.detach()))
        text_hidden = self.text_transformer(text_embeddings)
        video_pooled = video_hidden.mean(dim=1)
        text_pooled = text_hidden.mean(dim=1)
        logits_per_video = self.similarity(video_pooled, text_pooled, self.temperature)
        logits_per_text = logits_per_video.transpose(0, 1)
        targets = torch.arange(logits_per_video.shape[0], device=logits_per_video.device)
        loss = F.cross_entropy(logits_per_video, targets, reduction="sum") + F.cross_entropy(
            logits_per_text, targets, reduction="sum"
        )
        return {
            "video_embeddings": video_pooled,
            "text_embeddings": text_pooled,
            "logits_per_video": logits_per_video,
            "logits_per_text": logits_per_text,
            "loss": loss,
        }
