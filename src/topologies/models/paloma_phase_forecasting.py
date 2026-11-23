import math

import torch
from torch import nn


class PalomaPhaseForecasting(nn.Module):
    architecture_id = "paloma-phase-forecasting"
    PRESETS = {
        "tiny": {
            "seq_len": 8,
            "pred_len": 4,
            "n_channels": 2,
            "cycle_len": 4,
            "hidden_dim": 8,
            "dropout": 0.0,
            "use_revin": True,
        },
        "small": {
            "seq_len": 96,
            "pred_len": 96,
            "n_channels": 7,
            "cycle_len": 24,
            "hidden_dim": 64,
            "dropout": 0.1,
            "use_revin": True,
        },
        "base": {
            "seq_len": 96,
            "pred_len": 96,
            "n_channels": 7,
            "cycle_len": 24,
            "hidden_dim": 256,
            "dropout": 0.5,
            "use_revin": True,
        },
    }

    def __init__(
        self,
        seq_len=96,
        pred_len=96,
        n_channels=7,
        cycle_len=24,
        hidden_dim=256,
        dropout=0.5,
        use_revin=True,
    ):
        super().__init__()
        for name, value in (
            ("seq_len", seq_len),
            ("pred_len", pred_len),
            ("n_channels", n_channels),
            ("cycle_len", cycle_len),
            ("hidden_dim", hidden_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise self._value_error("dropout must be a finite number in [0, 1)")
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be a finite number in [0, 1)")
        if not isinstance(use_revin, bool):
            raise self._value_error("use_revin must be a boolean")

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_channels = n_channels
        self.cycle_len = cycle_len
        self.hidden_dim = hidden_dim
        self.use_revin = use_revin

        self.Q = nn.Parameter(torch.randn(cycle_len, n_channels))
        self.P = nn.Parameter(torch.randn(cycle_len, n_channels))
        self.A = nn.Parameter(torch.randn(cycle_len, n_channels))

        self.phase_cos = nn.Linear(seq_len, hidden_dim)
        self.phase_sin = nn.Linear(seq_len, hidden_dim)
        self.phase_angle = nn.Linear(seq_len, hidden_dim)
        self.phase_out = nn.Linear(hidden_dim, hidden_dim)
        self.phase_residual = nn.Linear(seq_len, hidden_dim)
        self.phase_norm = nn.LayerNorm(hidden_dim)

        self.amplitude_content = nn.Linear(seq_len, hidden_dim)
        self.amplitude_scale = nn.Linear(seq_len, hidden_dim)
        self.amplitude_shift = nn.Linear(seq_len, hidden_dim)
        self.amplitude_out = nn.Linear(hidden_dim, hidden_dim)

        self.input_proj = nn.Linear(seq_len, hidden_dim)
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, pred_len)

    @staticmethod
    def _value_error(message):
        return ValueError(message)

    @staticmethod
    def _phase_indices(cycle_index, seq_len, cycle_len):
        offsets = torch.arange(seq_len, device=cycle_index.device).unsqueeze(0)
        return (cycle_index.reshape(-1, 1) + offsets) % cycle_len

    @staticmethod
    def _gram_schmidt(h_cos, h_sin):
        projection = (h_sin * h_cos).sum(dim=-1, keepdim=True) / (
            h_cos.square().sum(dim=-1, keepdim=True) + 1e-6
        )
        return h_sin - projection * h_cos

    @staticmethod
    def _rotate(h_cos, h_sin_perp, theta):
        return torch.cos(theta) * h_cos + torch.sin(theta) * h_sin_perp

    @staticmethod
    def _amplitude_affine(content, scale, shift):
        return scale * content + shift

    @staticmethod
    def _normalize_instance(x):
        mean = x.mean(dim=1, keepdim=True)
        scale = torch.sqrt(
            x.var(dim=1, keepdim=True, unbiased=x.shape[1] > 1) + 1e-5
        )
        return (x - mean) / scale, mean, scale

    @staticmethod
    def _denormalize_instance(x, mean, scale):
        return x * scale + mean

    def _validate_inputs(self, x, cycle_index):
        if not isinstance(x, torch.Tensor):
            raise self._value_error("x must be a torch.Tensor with rank 3 [B, T, N]")
        if x.ndim != 3:
            raise ValueError("x must have rank 3 [B, T, N]")
        if not x.is_floating_point():
            raise ValueError("x must have a floating-point dtype")
        if x.shape[1] != self.seq_len:
            raise ValueError(f"x sequence length must match seq_len={self.seq_len}")
        if x.shape[2] != self.n_channels:
            raise ValueError(f"x channel count must match n_channels={self.n_channels}")
        if x.device != self.Q.device:
            raise ValueError("x must be on the same device as the model")
        if not isinstance(cycle_index, torch.Tensor):
            raise self._value_error("cycle_index must be an integer tensor with shape [B]")
        if cycle_index.ndim != 1:
            raise ValueError("cycle_index must have rank 1 and shape [B]")
        if cycle_index.shape[0] != x.shape[0]:
            raise ValueError("cycle_index batch size must match x batch size")
        if cycle_index.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError("cycle_index must have an integer dtype")
        if cycle_index.device != x.device:
            raise ValueError("cycle_index and x must be on the same device")

    def forward(self, x, cycle_index):
        self._validate_inputs(x, cycle_index)
        if self.use_revin:
            x_norm, mean, scale = self._normalize_instance(x)
        else:
            x_norm, mean, scale = x, None, None

        x_channel_first = x_norm.permute(0, 2, 1)
        indices = self._phase_indices(cycle_index, self.seq_len, self.cycle_len)
        q = self.Q[indices].permute(0, 2, 1)
        phase = self.P[indices].permute(0, 2, 1)
        amplitude = self.A[indices].permute(0, 2, 1)
        residual = x_channel_first - q

        h_cos = self.phase_cos(residual)
        h_sin = self.phase_sin(residual)
        theta = torch.pi * torch.tanh(self.phase_angle(phase))
        h_sin_perp = self._gram_schmidt(h_cos, h_sin)
        rotated = self._rotate(h_cos, h_sin_perp, theta)
        phase_hidden = self.phase_norm(
            self.phase_out(rotated) + self.phase_residual(residual)
        )

        content = self.amplitude_content(residual)
        scale_control = torch.sigmoid(self.amplitude_scale(amplitude))
        shift_control = self.amplitude_shift(amplitude)
        amplitude_hidden = self.amplitude_out(
            self._amplitude_affine(content, scale_control, shift_control)
        )

        hidden = torch.nn.functional.gelu(self.hidden_proj(phase_hidden + amplitude_hidden))
        input_embedding = self.input_proj(x_channel_first)
        forecast = self.output_proj(self.dropout(hidden + input_embedding)).permute(0, 2, 1)
        if self.use_revin:
            forecast = self._denormalize_instance(forecast, mean, scale)
        return forecast
