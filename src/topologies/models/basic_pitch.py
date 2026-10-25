import math

import torch
from torch import nn


class BasicPitch(nn.Module):
    architecture_id = "basic-pitch"
    PRESETS = {
        "tiny": {"input_features": 8, "num_pitches": 6, "hidden_features": 8, "threshold": 0.5},
        "small": {"input_features": 8, "num_pitches": 24, "hidden_features": 32, "threshold": 0.5},
    }

    def __init__(self, input_features=8, num_pitches=6, hidden_features=8, threshold=0.5):
        super().__init__()
        for name, value in (
            ("input_features", input_features),
            ("num_pitches", num_pitches),
            ("hidden_features", hidden_features),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not 0.0 <= threshold <= 1.0
            or not math.isfinite(threshold)
        ):
            raise ValueError("threshold must be a finite number between zero and one")
        self.input_features = input_features
        self.encoder = nn.Sequential(nn.Linear(input_features, hidden_features), nn.Tanh())
        self.onset_head = nn.Linear(hidden_features, 1)
        self.pitch_head = nn.Linear(hidden_features, num_pitches)
        self.threshold = float(threshold)

    def forward(self, frames, threshold=None):
        if not isinstance(frames, torch.Tensor) or frames.ndim != 3:
            raise ValueError("frames must be a tensor with shape (batch, frames, features)")
        if frames.shape[0] < 1 or frames.shape[1] < 1:
            raise ValueError("frames must contain at least one batch item and one frame")
        if frames.shape[-1] != self.input_features:
            raise ValueError(f"frames must have input_features={self.input_features}")
        cutoff = self.threshold if threshold is None else threshold
        if (
            isinstance(cutoff, bool)
            or not isinstance(cutoff, (int, float))
            or not 0.0 <= cutoff <= 1.0
            or not math.isfinite(cutoff)
        ):
            raise ValueError("threshold must be a finite number between zero and one")
        hidden = self.encoder(frames)
        onset_probability = torch.sigmoid(self.onset_head(hidden).squeeze(-1))
        pitch_probability = torch.sigmoid(self.pitch_head(hidden))
        notes = (onset_probability.unsqueeze(-1) >= cutoff) & (pitch_probability >= cutoff)
        return {
            "onset_probability": onset_probability,
            "pitch_probability": pitch_probability,
            "notes": notes,
        }
