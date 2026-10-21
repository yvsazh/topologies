import torch
from torch import nn


def circular_frequency_context(spectrogram, num_neighbors):
    if spectrogram.ndim != 3:
        raise ValueError("spectrogram must have shape (batch, frames, frequencies)")
    if num_neighbors < 0:
        raise ValueError("num_neighbors cannot be negative")
    return torch.stack(
        [torch.roll(spectrogram, shifts=-offset, dims=-1)
         for offset in range(-num_neighbors, num_neighbors + 1)],
        dim=-1,
    )


def normalize_full_band(spectrogram):
    scale = spectrogram.mean(dim=(1, 2), keepdim=True).clamp_min(torch.finfo(spectrogram.dtype).eps)
    return spectrogram / scale


def normalize_sub_band(sub_input):
    scale = sub_input.mean(dim=(1, 3), keepdim=True).clamp_min(torch.finfo(sub_input.dtype).eps)
    return sub_input / scale


class FullSubNet(nn.Module):
    architecture_id = "fullsubnet"
    def __init__(self, num_frequencies, hidden_size=8, num_neighbors=1, sub_hidden_size=None):
        super().__init__()
        if num_frequencies < 1 or hidden_size < 1 or num_neighbors < 0:
            raise ValueError("frequency and hidden sizes must be positive; neighbors cannot be negative")
        if sub_hidden_size is None:
            sub_hidden_size = max(1, 3 * hidden_size // 4)
        if sub_hidden_size < 1:
            raise ValueError("sub_hidden_size must be positive")
        self.num_frequencies = num_frequencies
        self.num_neighbors = num_neighbors
        self.full_band = nn.LSTM(num_frequencies, hidden_size, num_layers=2, batch_first=True)
        self.full_projection = nn.Linear(hidden_size, num_frequencies)
        self.sub_band = nn.LSTM(2 * num_neighbors + 2, sub_hidden_size, num_layers=2, batch_first=True)
        self.cirm_projection = nn.Linear(sub_hidden_size, 2)

    def forward(self, spectrogram):
        if spectrogram.ndim != 3 or spectrogram.shape[-1] != self.num_frequencies:
            raise ValueError("spectrogram must have shape (batch, frames, num_frequencies)")
        normalized_full = normalize_full_band(spectrogram)
        full_sequence, _ = self.full_band(normalized_full)
        full_features = torch.relu(self.full_projection(full_sequence))
        contexts = circular_frequency_context(spectrogram, self.num_neighbors)
        sub_input = torch.cat((contexts, full_features.unsqueeze(-1)), dim=-1)
        normalized_sub = normalize_sub_band(sub_input)
        sub_input = normalized_sub.permute(0, 2, 1, 3).reshape(
            -1, spectrogram.shape[1], 2 * self.num_neighbors + 2
        )
        sub_features, _ = self.sub_band(sub_input)
        cirm = self.cirm_projection(sub_features)
        return cirm.reshape(
            spectrogram.shape[0], self.num_frequencies, spectrogram.shape[1], 2
        ).transpose(1, 2)
