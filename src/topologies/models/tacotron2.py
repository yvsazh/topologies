import torch
from torch import nn


class Tacotron2(nn.Module):
    architecture_id = "tacotron-2"
    PRESETS = {"tiny": {"encoder_channels": 8, "decoder_channels": 4}}

    def __init__(self, encoder_channels=8, decoder_channels=4, attention_channels=8, location_filters=4, location_kernel_size=3):
        super().__init__()
        if min(encoder_channels, decoder_channels, attention_channels, location_filters) <= 0:
            raise ValueError("channel dimensions must be positive")
        if location_kernel_size <= 0 or location_kernel_size % 2 == 0:
            raise ValueError("location_kernel_size must be a positive odd integer")
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.encoder_projection = nn.Linear(encoder_channels, attention_channels, bias=False)
        self.decoder_projection = nn.Linear(decoder_channels, attention_channels, bias=False)
        self.location_convolution = nn.Conv1d(2, location_filters, location_kernel_size, padding=(location_kernel_size - 1) // 2, bias=False)
        self.location_projection = nn.Linear(location_filters, attention_channels, bias=False)
        self.energy_projection = nn.Linear(attention_channels, 1, bias=False)
        self.mel_projection = nn.Linear(decoder_channels + encoder_channels, encoder_channels)
        self.register_buffer("attention_history", torch.empty(0), persistent=False)
        self.register_buffer("previous_attention", torch.empty(0), persistent=False)

    def reset_attention(self):
        self.attention_history = torch.empty(0, device=self.energy_projection.weight.device)
        self.previous_attention = torch.empty(0, device=self.energy_projection.weight.device)

    def attention(self, encoder_states, decoder_state, previous_attention=None, cumulative_attention=None):
        if encoder_states.ndim != 3 or decoder_state.ndim != 2:
            raise ValueError("encoder states must be (batch, time, channels), decoder state must be (batch, channels)")
        batch, steps, _ = encoder_states.shape
        if (encoder_states.shape[2] != self.encoder_channels or
                decoder_state.shape != (batch, self.decoder_channels) or steps < 1):
            raise ValueError("encoder or decoder channel dimensions do not match the constructor")
        if previous_attention is None:
            if self.previous_attention.numel() == 0 or self.previous_attention.shape != (batch, steps):
                previous_attention = encoder_states.new_zeros(batch, steps)
            else:
                previous_attention = self.previous_attention.to(device=encoder_states.device, dtype=encoder_states.dtype)
        if previous_attention.shape != (batch, steps):
            raise ValueError("previous attention must have shape (batch, time)")
        if cumulative_attention is None:
            if self.attention_history.numel() == 0 or self.attention_history.shape != (batch, steps):
                cumulative_attention = encoder_states.new_zeros(batch, steps)
            else:
                cumulative_attention = self.attention_history.to(device=encoder_states.device, dtype=encoder_states.dtype)
        if cumulative_attention.shape != (batch, steps):
            raise ValueError("cumulative attention must have shape (batch, time)")
        location_features = self.location_convolution(
            torch.stack((previous_attention, cumulative_attention), dim=1)
        ).transpose(1, 2)
        energies = self.energy_projection(torch.tanh(self.encoder_projection(encoder_states) + self.decoder_projection(decoder_state).unsqueeze(1) + self.location_projection(location_features))).squeeze(-1)
        weights = torch.softmax(energies, dim=-1)
        self.previous_attention = weights.detach()
        self.attention_history = (cumulative_attention + weights).detach()
        context = torch.bmm(weights.unsqueeze(1), encoder_states).squeeze(1)
        return context, weights, energies

    def forward(self, encoder_states, decoder_state, previous_attention=None):
        context, weights, energies = self.attention(encoder_states, decoder_state, previous_attention)
        output = self.mel_projection(torch.cat((decoder_state, context), dim=-1))
        return {"output": output, "attention": weights, "energies": energies, "context": context}
