import torch
from torch import nn


class FNO(nn.Module):
    architecture_id = "fno"
    component_id = "fno"

    def __init__(self, in_channels=1, out_channels=None, modes=8):
        super().__init__()
        if in_channels <= 0 or modes <= 0 or (out_channels is not None and out_channels <= 0):
            raise ValueError("channel counts and modes must be positive")
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.spectral_weight = nn.Parameter(torch.empty(modes, in_channels, out_channels, dtype=torch.cfloat))
        self.skip = nn.Linear(in_channels, out_channels)
        nn.init.normal_(self.spectral_weight, std=1 / (in_channels * out_channels) ** 0.5)

    def forward(self, values):
        if not torch.is_tensor(values):
            values = torch.as_tensor(values, dtype=self.skip.weight.dtype, device=self.skip.weight.device)
        if values.ndim != 3 or values.shape[-1] != self.in_channels:
            raise ValueError("values must have shape (batch, length, in_channels)")
        length = values.shape[1]
        if length < 1:
            raise ValueError("length must be positive")
        spectrum = torch.fft.rfft(values, dim=1)
        retained = min(self.modes, spectrum.shape[1])
        filtered = torch.zeros(
            values.shape[0], spectrum.shape[1], self.out_channels,
            dtype=spectrum.dtype, device=spectrum.device,
        )
        filtered[:, :retained] = torch.einsum(
            "bmi,mio->bmo", spectrum[:, :retained], self.spectral_weight[:retained]
        )
        return torch.fft.irfft(filtered, n=length, dim=1) + self.skip(values)
