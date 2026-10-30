import math
import numbers

import torch
from torch import nn


class _SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, normalized_voltage, dampening):
        ctx.save_for_backward(normalized_voltage)
        ctx.dampening = dampening
        return (normalized_voltage > 0).to(normalized_voltage.dtype)

    @staticmethod
    def backward(ctx, gradient):
        (normalized_voltage,) = ctx.saved_tensors
        slope = (1.0 - normalized_voltage.abs()).clamp_min(0.0)
        return gradient * ctx.dampening * slope, None


class ALIF(nn.Module):
    component_id = "alif"

    def __init__(
        self,
        baseline_threshold,
        adaptation_scale,
        membrane_time_constant,
        adaptation_time_constant,
        refractory_steps,
        dt=1.0,
        surrogate_dampening=0.3,
    ):
        super().__init__()
        baseline = self._parameter_tensor("baseline_threshold", baseline_threshold)
        adaptation = self._parameter_tensor("adaptation_scale", adaptation_scale)
        membrane_tau = self._parameter_tensor(
            "membrane_time_constant", membrane_time_constant, positive=True
        )
        adaptation_tau = self._parameter_tensor(
            "adaptation_time_constant", adaptation_time_constant, positive=True
        )
        if torch.any(baseline <= 0):
            raise ValueError("baseline_threshold must contain only positive values")
        if torch.any(adaptation < 0):
            raise ValueError("adaptation_scale must contain only nonnegative values")
        lengths = {
            baseline.numel(),
            adaptation.numel(),
            membrane_tau.numel(),
            adaptation_tau.numel(),
        }
        width = max(lengths)
        if any(length not in (1, width) for length in lengths):
            raise ValueError("per-neuron parameters must have matching lengths")
        if isinstance(refractory_steps, bool) or not isinstance(
            refractory_steps, numbers.Integral
        ):
            self._raise_value_error("refractory_steps must be a nonnegative integer")
        if refractory_steps < 0:
            raise ValueError("refractory_steps must be a nonnegative integer")
        if isinstance(dt, bool) or not isinstance(dt, numbers.Real):
            self._raise_value_error("dt must be a positive finite number")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be a positive finite number")
        if isinstance(surrogate_dampening, bool) or not isinstance(
            surrogate_dampening, numbers.Real
        ):
            self._raise_value_error("surrogate_dampening must be in [0, 1)")
        if not math.isfinite(surrogate_dampening) or not 0 <= surrogate_dampening < 1:
            raise ValueError("surrogate_dampening must be in [0, 1)")
        self.register_buffer("baseline_threshold", baseline)
        self.register_buffer("adaptation_scale", adaptation)
        self.register_buffer("membrane_time_constant", membrane_tau)
        self.register_buffer("adaptation_time_constant", adaptation_tau)
        self.refractory_steps = int(refractory_steps)
        self.dt = float(dt)
        self.surrogate_dampening = float(surrogate_dampening)

    @staticmethod
    def _raise_value_error(message):
        raise ValueError(message)

    @staticmethod
    def _parameter_tensor(name, value, positive=False):
        if isinstance(value, bool):
            ALIF._raise_value_error(
                f"{name} must be a finite scalar or one-dimensional sequence"
            )
        try:
            tensor = torch.as_tensor(value, dtype=torch.get_default_dtype())
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError(
                f"{name} must be a finite scalar or one-dimensional sequence"
            ) from error
        if tensor.ndim > 1 or tensor.numel() == 0:
            raise ValueError(f"{name} must be a finite scalar or one-dimensional sequence")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must contain only finite values")
        if positive and torch.any(tensor <= 0):
            raise ValueError(f"{name} must contain only positive values")
        return tensor.reshape(-1)

    def _parameter_for_input(self, parameter, neurons, x, name):
        if parameter.numel() not in (1, neurons):
            raise ValueError(
                f"{name} must be scalar or contain one value per input neuron"
            )
        parameter = parameter.to(device=x.device, dtype=x.dtype)
        return parameter.expand(neurons).reshape(1, neurons)

    @staticmethod
    def _initial_state(initial_state, batch, neurons, x):
        shape = (batch, neurons)
        if initial_state is None:
            return (
                x.new_zeros(shape),
                x.new_zeros(shape),
                torch.zeros(shape, dtype=torch.long, device=x.device),
            )
        if not isinstance(initial_state, (tuple, list)) or len(initial_state) != 3:
            raise ValueError("state must contain membrane, adaptation, and refractory tensors")
        membrane, adaptation, refractory = initial_state
        if not all(torch.is_tensor(value) for value in initial_state):
            raise ValueError("state entries must be tensors")
        if any(value.shape != shape for value in initial_state):
            raise ValueError(f"state tensors must all have shape {shape}")
        membrane = membrane.to(device=x.device, dtype=x.dtype)
        adaptation = adaptation.to(device=x.device, dtype=x.dtype)
        refractory = refractory.to(device=x.device)
        if not torch.isfinite(membrane).all() or not torch.isfinite(adaptation).all():
            raise ValueError("state membrane and adaptation values must be finite")
        if torch.is_floating_point(refractory) and (
            not torch.isfinite(refractory).all()
            or torch.any(
                refractory != refractory.round()
            )
        ):
            raise ValueError("state refractory values must be nonnegative integers")
        if torch.any(refractory < 0):
            raise ValueError("state refractory values must be nonnegative integers")
        return membrane, adaptation, refractory.to(dtype=torch.long)

    def forward(self, x, initial_state=None, return_state=False):
        if not torch.is_tensor(x) or not torch.is_floating_point(x):
            raise ValueError("input must be a floating-point tensor with shape [B, N] or [B, T, N]")
        if x.ndim == 2:
            sequence = x.unsqueeze(1)
            squeeze_time = True
        elif x.ndim == 3:
            sequence = x
            squeeze_time = False
        else:
            raise ValueError("input must have shape [B, N] or [B, T, N]")
        batch, steps, neurons = sequence.shape
        if batch < 1 or steps < 1 or neurons < 1:
            raise ValueError("input batch, sequence, and neuron dimensions must be nonempty")
        if not torch.isfinite(sequence).all():
            raise ValueError("input values must be finite")
        membrane, adaptation, refractory = self._initial_state(
            initial_state, batch, neurons, sequence
        )
        baseline = self._parameter_for_input(
            self.baseline_threshold, neurons, sequence, "baseline_threshold"
        )
        adaptation_scale = self._parameter_for_input(
            self.adaptation_scale, neurons, sequence, "adaptation_scale"
        )
        membrane_tau = self._parameter_for_input(
            self.membrane_time_constant, neurons, sequence, "membrane_time_constant"
        )
        adaptation_tau = self._parameter_for_input(
            self.adaptation_time_constant,
            neurons,
            sequence,
            "adaptation_time_constant",
        )
        alpha = torch.exp(-self.dt / membrane_tau)
        rho = torch.exp(-self.dt / adaptation_tau)
        spike_steps = []
        for drive in sequence.unbind(dim=1):
            threshold = baseline + adaptation_scale * adaptation
            normalized_voltage = (membrane - threshold) / threshold
            spike = _SurrogateSpike.apply(
                normalized_voltage, self.surrogate_dampening
            )
            eligible = refractory == 0
            spike = torch.where(eligible, spike, torch.zeros_like(spike))
            membrane = (
                alpha * membrane
                + (1.0 - alpha) * drive
                - threshold * spike * self.dt
            )
            adaptation = rho * adaptation + (1.0 - rho) * spike
            refractory = torch.where(
                eligible,
                torch.where(
                    spike.detach() > 0,
                    torch.full_like(refractory, self.refractory_steps),
                    refractory,
                ),
                (refractory - 1).clamp_min(0),
            )
            spike_steps.append(spike)
        spikes = torch.stack(spike_steps, dim=1)
        if squeeze_time:
            spikes = spikes.squeeze(1)
        if return_state:
            return spikes, (membrane, adaptation, refractory)
        return spikes
