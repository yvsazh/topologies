import torch
from torch import nn


class FastPitch(nn.Module):
    architecture_id = "fastpitch"
    def __init__(self, hidden_dim=8, n_mels=4, pitch_bins=16):
        super().__init__()
        if hidden_dim <= 0 or n_mels <= 0 or pitch_bins < 2:
            raise ValueError("hidden_dim and n_mels must be positive and pitch_bins at least two")
        self.hidden_dim = hidden_dim
        self.n_mels = n_mels
        self.pitch_bins = pitch_bins
        self.pitch_predictor = nn.Linear(hidden_dim, 1)
        self.pitch_embedding = nn.Embedding(pitch_bins, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, n_mels)

    def forward(self, token_states, durations, pitch=None):
        if token_states.ndim != 3:
            raise ValueError("token_states must have shape (batch, tokens, channels)")
        if durations.ndim != 2 or durations.shape != token_states.shape[:2]:
            raise ValueError("durations must have shape (batch, tokens)")
        if token_states.shape[-1] != self.hidden_dim:
            raise ValueError("token state channels do not match hidden_dim")
        if torch.any(durations < 0) or torch.any(durations != durations.round()):
            raise ValueError("durations must be nonnegative integers")
        pitch_values = self.pitch_predictor(token_states).squeeze(-1) if pitch is None else pitch
        if pitch_values.shape != token_states.shape[:2]:
            raise ValueError("pitch must have shape (batch, tokens)")
        pitch_indices = (pitch_values.clamp(0, 1) * (self.pitch_bins - 1)).round().long()
        conditioned = token_states + self.pitch_embedding(pitch_indices)
        outputs = []
        for states, token_durations in zip(conditioned, durations):
            expanded = torch.repeat_interleave(states, token_durations.to(torch.long), dim=0)
            outputs.append(self.decoder(expanded))
        lengths = [item.shape[0] for item in outputs]
        if len(set(lengths)) > 1:
            raise ValueError("all batch items must have the same expanded length")
        if not outputs:
            return token_states.new_empty((0, 0, self.n_mels))
        return torch.stack(outputs)
