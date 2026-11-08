import math
import numbers

import torch
from torch import nn


class LIFNeuron(nn.Module):
    component_id = "lif-neuron"

    def __init__(
        self,
        threshold=15.0,
        membrane_time_constant=20.0,
        dt=0.1,
        reset_potential=0.0,
        refractory_steps=10,
    ):
        super().__init__()
        self.threshold = self._finite_real("threshold", threshold)
        self.membrane_time_constant = self._finite_real(
            "membrane_time_constant", membrane_time_constant
        )
        self.dt = self._finite_real("dt", dt)
        self.reset_potential = self._finite_real("reset_potential", reset_potential)
        if self.membrane_time_constant <= 0:
            raise ValueError("membrane_time_constant must be positive")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.threshold <= self.reset_potential:
            raise ValueError("threshold must be greater than reset_potential")
        if isinstance(refractory_steps, bool) or not isinstance(
            refractory_steps, numbers.Integral
        ):
            self._raise_value_error("refractory_steps must be a nonnegative integer")
        if refractory_steps < 0:
            raise ValueError("refractory_steps must be a nonnegative integer")
        self.refractory_steps = int(refractory_steps)

    @staticmethod
    def _raise_value_error(message):
        raise ValueError(message)

    @staticmethod
    def _finite_real(name, value):
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            LIFNeuron._raise_value_error(f"{name} must be a finite real number")
        try:
            value = float(value)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a finite real number") from error
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite real number")
        return value

    @staticmethod
    def _initial_state(initial_state, batch, neurons, x, reset_potential):
        shape = (batch, neurons)
        if initial_state is None:
            return (
                x.new_full(shape, reset_potential),
                torch.zeros(shape, dtype=torch.long, device=x.device),
            )
        if not isinstance(initial_state, (tuple, list)) or len(initial_state) != 2:
            raise ValueError("initial_state must contain membrane and refractory_remaining tensors")
        membrane, refractory_remaining = initial_state
        if not torch.is_tensor(membrane) or not torch.is_tensor(refractory_remaining):
            raise ValueError("initial_state entries must be tensors")
        if membrane.shape != shape or refractory_remaining.shape != shape:
            raise ValueError(f"initial_state tensors must both have shape {shape}")
        if not torch.is_floating_point(membrane) or not torch.isfinite(membrane).all():
            raise ValueError("initial membrane must contain finite floating-point values")
        if refractory_remaining.is_complex() or refractory_remaining.dtype == torch.bool:
            raise ValueError("refractory_remaining must contain nonnegative integer values")
        if torch.is_floating_point(refractory_remaining) and (
            not torch.isfinite(refractory_remaining).all()
            or torch.any(refractory_remaining != refractory_remaining.round())
        ):
            raise ValueError("refractory_remaining must contain nonnegative integer values")
        if torch.any(refractory_remaining < 0):
            raise ValueError("refractory_remaining must contain nonnegative integer values")
        return (
            membrane.to(device=x.device, dtype=x.dtype),
            refractory_remaining.to(device=x.device, dtype=torch.long),
        )

    def forward(
        self,
        input_jump,
        initial_state=None,
        return_state=False,
        return_membrane=False,
    ):
        if return_state and return_membrane:
            raise ValueError("return_state and return_membrane cannot both be enabled")
        if not torch.is_tensor(input_jump) or not torch.is_floating_point(input_jump):
            raise ValueError("input_jump must be a floating-point tensor with shape [B, N] or [B, T, N]")
        if input_jump.ndim == 2:
            sequence = input_jump.unsqueeze(1)
            squeeze_time = True
        elif input_jump.ndim == 3:
            sequence = input_jump
            squeeze_time = False
        else:
            raise ValueError("input_jump must have shape [B, N] or [B, T, N]")
        batch, steps, neurons = sequence.shape
        if batch < 1 or steps < 1 or neurons < 1:
            raise ValueError("input_jump batch, sequence, and neuron dimensions must be nonempty")
        if not torch.isfinite(sequence).all():
            raise ValueError("input_jump values must be finite")
        membrane, refractory_remaining = self._initial_state(
            initial_state, batch, neurons, sequence, self.reset_potential
        )
        decay = sequence.new_tensor(math.exp(-self.dt / self.membrane_time_constant))
        threshold = sequence.new_tensor(self.threshold)
        reset = sequence.new_tensor(self.reset_potential)
        spike_steps = []
        for drive in sequence.unbind(dim=1):
            eligible = refractory_remaining == 0
            voltage = decay * membrane + drive
            firing = eligible & (voltage >= threshold)
            spikes = firing.to(dtype=sequence.dtype)
            membrane = torch.where(
                eligible,
                torch.where(firing, reset, voltage),
                reset,
            )
            refractory_remaining = torch.where(
                firing,
                torch.full_like(refractory_remaining, self.refractory_steps),
                torch.where(
                    eligible,
                    torch.zeros_like(refractory_remaining),
                    refractory_remaining - 1,
                ),
            )
            spike_steps.append(spikes)
        if return_membrane:
            return membrane
        spikes = torch.stack(spike_steps, dim=1)
        if squeeze_time:
            spikes = spikes.squeeze(1)
        if return_state:
            return spikes, (membrane, refractory_remaining)
        return spikes
