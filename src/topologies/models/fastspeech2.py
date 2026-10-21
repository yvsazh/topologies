import torch
from torch import nn


class FastSpeech2(nn.Module):
    architecture_id = "fastspeech-2"
    def __init__(self, hidden_dim=8, n_mels=4, pitch_bins=32, energy_bins=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_mels = n_mels
        self.pitch_bins = pitch_bins
        self.energy_bins = energy_bins
        self.pitch_embedding = nn.Embedding(pitch_bins, hidden_dim)
        self.energy_embedding = nn.Embedding(energy_bins, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, n_mels)

    def forward(self, token_states, durations, pitch, energy):
        if token_states.ndim != 3:
            raise ValueError("token_states must have shape (batch, tokens, channels)")
        if durations.ndim != 2 or durations.shape != token_states.shape[:2]:
            raise ValueError("durations must have shape (batch, tokens)")
        if token_states.shape[-1] != self.hidden_dim:
            raise ValueError("token state channels do not match hidden_dim")
        if torch.any(durations < 0) or torch.any(durations != durations.round()):
            raise ValueError("durations must be nonnegative integers")
        batch_frames = []
        for batch_index, (states, token_durations) in enumerate(zip(token_states, durations)):
            expanded = torch.repeat_interleave(states, token_durations.to(torch.long), dim=0)
            frame_count = expanded.shape[0]
            pitch_frames = pitch[batch_index].reshape(-1)
            energy_frames = energy[batch_index].reshape(-1)
            if pitch_frames.numel() != frame_count or energy_frames.numel() != frame_count:
                raise ValueError("pitch and energy must provide one value per expanded frame")
            if torch.any(pitch_frames < 0) or torch.any(pitch_frames >= self.pitch_bins):
                raise ValueError("pitch values must be valid embedding indices")
            if torch.any(energy_frames < 0) or torch.any(energy_frames >= self.energy_bins):
                raise ValueError("energy values must be valid embedding indices")
            conditioned = expanded + self.pitch_embedding(pitch_frames.long()) + self.energy_embedding(energy_frames.long())
            batch_frames.append(self.decoder(conditioned))
        lengths = [frames.shape[0] for frames in batch_frames]
        if len(set(lengths)) > 1:
            raise ValueError("all batch items must have the same expanded length")
        if not batch_frames:
            return token_states.new_empty((0, 0, self.n_mels))
        return torch.stack(batch_frames)
