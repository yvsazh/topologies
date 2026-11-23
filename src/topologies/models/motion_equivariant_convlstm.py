import math

import torch
from torch import nn


class MotionEquivariantConvLSTM(nn.Module):
    architecture_id = "motion-equivariant-convlstm"
    PRESETS = {
        "tiny": {
            "input_channels": 1,
            "hidden_channels": 8,
            "n_slots": 2,
            "kernel_size": 3,
            "bias": True,
            "phase_correlation_eps": 1e-8,
        },
        "small": {
            "input_channels": 1,
            "hidden_channels": 32,
            "n_slots": 2,
            "kernel_size": 3,
            "bias": True,
            "phase_correlation_eps": 1e-8,
        },
        "base": {
            "input_channels": 1,
            "hidden_channels": 64,
            "n_slots": 2,
            "kernel_size": 3,
            "bias": True,
            "phase_correlation_eps": 1e-8,
        },
    }

    def __init__(
        self,
        input_channels,
        hidden_channels,
        n_slots=2,
        kernel_size=3,
        bias=True,
        phase_correlation_eps=1e-8,
    ):
        super().__init__()
        dimensions = {
            "input_channels": input_channels,
            "hidden_channels": hidden_channels,
            "n_slots": n_slots,
        }
        for name, value in dimensions.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(kernel_size, int) or isinstance(kernel_size, bool) or kernel_size <= 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if bias is not True and bias is not False:
            raise ValueError("bias must be a boolean")
        if (
            not isinstance(phase_correlation_eps, (int, float))
            or isinstance(phase_correlation_eps, bool)
            or not math.isfinite(phase_correlation_eps)
            or phase_correlation_eps <= 0
        ):
            raise ValueError("phase_correlation_eps must be a finite positive number")

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.n_slots = n_slots
        self.kernel_size = kernel_size
        self.phase_correlation_eps = float(phase_correlation_eps)
        padding = kernel_size // 2
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding,
            padding_mode="circular",
            bias=bias,
        )
        if bias:
            with torch.no_grad():
                self.gates.bias[hidden_channels : 2 * hidden_channels].fill_(1.0)

    def _phase_correlation(self, first, second, n_peaks=1):
        if first.ndim != 4 or second.ndim != 4:
            raise ValueError("phase-correlation inputs must have shape [batch, channels, height, width]")
        if first.shape[0] != second.shape[0] or first.shape[-2:] != second.shape[-2:]:
            raise ValueError("phase-correlation inputs must have matching batch and spatial dimensions")
        if not isinstance(n_peaks, int) or isinstance(n_peaks, bool) or n_peaks <= 0:
            raise ValueError("n_peaks must be a positive integer")

        batch, _, height, width = first.shape
        if n_peaks > height * width:
            raise ValueError("n_peaks cannot exceed the number of spatial positions")
        first_spectrum = torch.fft.rfft2(first.mean(dim=1), dim=(-2, -1))
        second_spectrum = torch.fft.rfft2(second.mean(dim=1), dim=(-2, -1))
        cross_power = first_spectrum * torch.conj(second_spectrum)
        cross_power = cross_power / (cross_power.abs() + self.phase_correlation_eps)
        correlation = torch.fft.irfft2(cross_power, s=(height, width), dim=(-2, -1))
        peaks = correlation.reshape(batch, -1).topk(n_peaks, dim=1).indices
        dy = (peaks // width).to(first.dtype)
        dx = (peaks % width).to(first.dtype)
        dx = torch.where(dx > width / 2, dx - width, dx)
        dy = torch.where(dy > height / 2, dy - height, dy)
        return torch.stack((-dx, -dy), dim=-1)

    def _transport(self, state, velocity):
        if state.ndim != 5:
            raise ValueError("state must have shape [batch, slots, channels, height, width]")
        batch, slots, channels, height, width = state.shape
        if velocity.shape != (batch, slots, 2):
            raise ValueError("velocity must have shape [batch, slots, 2]")
        vx = velocity[..., 0].round().to(torch.long)
        vy = velocity[..., 1].round().to(torch.long)
        y = torch.arange(height, device=state.device).view(1, 1, height, 1)
        x = torch.arange(width, device=state.device).view(1, 1, 1, width)
        source_y = torch.remainder(y - vy[..., None, None], height)
        source_x = torch.remainder(x - vx[..., None, None], width)
        indices = (source_y * width + source_x).reshape(batch, slots, 1, height * width)
        indices = indices.expand(-1, -1, channels, -1)
        return torch.gather(state.flatten(-2), -1, indices).reshape(batch, slots, channels, height, width)

    def _cell_step(self, x, h, c, velocity):
        batch, slots, _, height, width = h.shape
        transported_h = self._transport(h, velocity)
        transported_c = self._transport(c, velocity)
        expanded_x = x.unsqueeze(1).expand(-1, slots, -1, -1, -1)
        gate_values = self.gates(
            torch.cat(
                (
                    expanded_x.reshape(batch * slots, self.input_channels, height, width),
                    transported_h.reshape(batch * slots, self.hidden_channels, height, width),
                ),
                dim=1,
            )
        )
        input_gate, forget_gate, output_gate, candidate = gate_values.chunk(4, dim=1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        output_gate = torch.sigmoid(output_gate)
        candidate = torch.tanh(candidate)
        next_c = forget_gate * transported_c.reshape(batch * slots, self.hidden_channels, height, width)
        next_c = next_c + input_gate * candidate
        next_h = output_gate * torch.tanh(next_c)
        return (
            next_h.reshape(batch, slots, self.hidden_channels, height, width),
            next_c.reshape(batch, slots, self.hidden_channels, height, width),
        )

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError("input must have shape [batch, time, channels, height, width]")
        batch, time, channels, height, width = x.shape
        if batch <= 0 or time <= 0 or height <= 0 or width <= 0:
            raise ValueError("input batch, time, height, and width dimensions must be positive")
        if channels != self.input_channels:
            raise ValueError(f"input channels must equal input_channels ({self.input_channels})")
        if not x.is_floating_point():
            raise ValueError("input must have a floating-point dtype")
        if time > 1 and self.n_slots > height * width:
            raise ValueError("n_slots cannot exceed the number of spatial positions for phase-correlation bootstrap")

        h = x.new_zeros(batch, self.n_slots, self.hidden_channels, height, width)
        c = torch.zeros_like(h)
        states = []
        velocity = x.new_zeros(batch, self.n_slots, 2)
        for step in range(time):
            if step == 1:
                velocity = self._phase_correlation(x[:, 0], x[:, 1], n_peaks=self.n_slots)
            elif step > 1:
                hidden_mean = h.mean(dim=2).reshape(batch * self.n_slots, 1, height, width)
                current = x[:, step].unsqueeze(1).expand(-1, self.n_slots, -1, -1, -1)
                current = current.reshape(batch * self.n_slots, channels, height, width)
                velocity = self._phase_correlation(hidden_mean, current, n_peaks=1).reshape(
                    batch, self.n_slots, 2
                )
            h, c = self._cell_step(x[:, step], h, c, velocity)
            states.append(h)
        return torch.stack(states, dim=1)
