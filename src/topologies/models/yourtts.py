import torch
from torch import nn


class YourTTS(nn.Module):
    architecture_id = "yourtts"
    def __init__(self, text_channels: int, speaker_channels: int):
        super().__init__()
        if text_channels <= 0 or speaker_channels <= 0:
            raise ValueError("text_channels and speaker_channels must be positive")
        self.text_channels = text_channels
        self.speaker_channels = speaker_channels
        self.speaker_projection = nn.Linear(speaker_channels, text_channels)

    def forward(self, text: torch.Tensor, speaker_embedding: torch.Tensor) -> torch.Tensor:
        if text.ndim != 3 or text.shape[-1] != self.text_channels:
            raise ValueError("text must have shape (batch, sequence, text_channels)")
        if speaker_embedding.ndim != 2 or speaker_embedding.shape != (text.shape[0], self.speaker_channels):
            raise ValueError("speaker_embedding must have shape (batch, speaker_channels)")
        projected_speaker = self.speaker_projection(speaker_embedding).unsqueeze(1)
        return text + projected_speaker
