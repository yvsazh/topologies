import torch
from torch import nn


class XTTS(nn.Module):
    architecture_id = "xtts"

    def __init__(self, speaker_channels=4, layer_channels=(8, 4)):
        super().__init__()
        if speaker_channels <= 0 or not layer_channels or any(width <= 0 for width in layer_channels):
            raise ValueError("speaker_channels and every layer channel count must be positive")
        self.speaker_channels = speaker_channels
        self.layer_channels = tuple(layer_channels)
        self.speaker_projections = nn.ModuleList(
            nn.Linear(speaker_channels, width) for width in self.layer_channels
        )

    def forward(self, upsampling_features, speaker_embedding):
        if torch.is_tensor(upsampling_features):
            if upsampling_features.ndim != 4:
                raise ValueError("packed upsampling_features must have shape (layers, batch, channels, time)")
            upsampling_features = upsampling_features.unbind(dim=0)
        if not isinstance(upsampling_features, (tuple, list)):
            raise TypeError("upsampling_features must be a sequence of layer tensors")
        if len(upsampling_features) != len(self.speaker_projections):
            raise ValueError("one feature tensor is required for each configured upsampling layer")
        if speaker_embedding.ndim != 2 or speaker_embedding.shape[1] != self.speaker_channels:
            raise ValueError("speaker_embedding must have shape (batch, speaker_channels)")

        conditioned = {}
        for index, (features, projection, channels) in enumerate(
            zip(upsampling_features, self.speaker_projections, self.layer_channels)
        ):
            if features.ndim != 3 or features.shape[0] != speaker_embedding.shape[0]:
                raise ValueError("each feature tensor must have shape (batch, channels, time)")
            if features.shape[1] != channels:
                raise ValueError(f"layer {index} expects {channels} channels")
            projected_speaker = projection(speaker_embedding).unsqueeze(-1)
            conditioned[f"layer_{index}"] = features + projected_speaker
        return conditioned
