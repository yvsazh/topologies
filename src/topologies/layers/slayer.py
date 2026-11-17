import math

import torch
from torch import nn
from torch.nn import functional as F


def _causal_filter(signal, kernel, dt):
    channels = signal.shape[1]
    if kernel.ndim == 1:
        kernels = kernel.unsqueeze(0).expand(channels, -1)
    else:
        kernels = kernel
    if kernels.shape[0] != channels or kernels.shape[1] == 0:
        raise ValueError("kernel must have one non-empty row per signal channel")
    weights = kernels.flip(-1).unsqueeze(1)
    padded = F.pad(signal, (kernels.shape[1] - 1, 0))
    return F.conv1d(padded, weights, groups=channels) * dt


def _shift_kernels(kernel, delays, length, dt):
    features = delays.numel()
    values = F.pad(kernel, (0, max(0, length - kernel.numel())))[:length]
    positions = torch.arange(length, device=kernel.device, dtype=kernel.dtype)
    source = positions.unsqueeze(0) - delays.unsqueeze(1) / dt
    lower = torch.floor(source).long()
    fraction = source - lower.to(source.dtype)
    upper = lower + 1
    lower_valid = (lower >= 0) & (lower < length)
    upper_valid = (upper >= 0) & (upper < length)
    lower_values = values[lower.clamp(0, max(length - 1, 0))] * lower_valid
    upper_values = values[upper.clamp(0, max(length - 1, 0))] * upper_valid
    shifted = lower_values * (1.0 - fraction) + upper_values * fraction
    if shifted.shape[0] != features:
        raise ValueError("delay count must match input feature count")
    return shifted


def _future_correlation(error, kernel):
    time = error.shape[-1]
    if kernel.ndim == 1:
        kernels = kernel.unsqueeze(0).expand(error.shape[-2], -1)
    else:
        kernels = kernel
    if kernels.shape[0] != error.shape[-2] or kernels.shape[1] == 0:
        raise ValueError("kernel must have one non-empty row per error channel")
    result = torch.zeros_like(error)
    for lag in range(min(time, kernels.shape[1])):
        result[..., : time - lag] += error[..., lag:] * kernels[:, lag].view(
            *([1] * (error.ndim - 2)), -1, 1
        )
    return result


def _time_derivative(signal, dt):
    if signal.shape[-1] < 2:
        return torch.zeros_like(signal)
    derivative = torch.empty_like(signal)
    derivative[..., 0] = (signal[..., 1] - signal[..., 0]) / dt
    derivative[..., -1] = (signal[..., -1] - signal[..., -2]) / dt
    if signal.shape[-1] > 2:
        derivative[..., 1:-1] = (signal[..., 2:] - signal[..., :-2]) / (2.0 * dt)
    return derivative


class _SLAYERFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, delays, epsilon, refractory, dt, threshold,
                rho_tau, rho_scale, learnable_delays):
        inputs = inputs.contiguous()
        weight_value = weight.to(device=inputs.device, dtype=inputs.dtype)
        delay_value = delays.to(device=inputs.device, dtype=inputs.dtype).clamp_min(0.0)
        epsilon_value = epsilon.to(device=inputs.device, dtype=inputs.dtype)
        refractory_value = refractory.to(device=inputs.device, dtype=inputs.dtype)
        shifted_epsilon = _shift_kernels(epsilon_value, delay_value, inputs.shape[-1], dt)
        activity = _causal_filter(inputs, shifted_epsilon, dt)
        membrane = torch.einsum("oi,bit->bot", weight_value, activity)
        spikes = torch.zeros_like(membrane)
        for time in range(inputs.shape[-1]):
            refractory_steps = min(time, refractory_value.numel() - 1)
            for lag in range(1, refractory_steps + 1):
                membrane[:, :, time] += dt * refractory_value[lag] * spikes[:, :, time - lag]
            spikes[:, :, time] = (membrane[:, :, time] >= threshold).to(inputs.dtype)
        ctx.save_for_backward(inputs, weight, delays, shifted_epsilon, activity, membrane)
        ctx.dt = dt
        ctx.threshold = threshold
        ctx.rho_tau = rho_tau
        ctx.rho_scale = rho_scale
        ctx.learnable_delays = learnable_delays
        return spikes, activity, membrane

    @staticmethod
    def backward(ctx, grad_spikes, grad_activity, grad_membrane):
        inputs, weight, delays, epsilon, activity, membrane = ctx.saved_tensors
        if grad_spikes is None:
            grad_spikes = torch.zeros_like(membrane)
        rho = SLAYERLayer.surrogate_derivative(
            membrane,
            threshold=ctx.threshold,
            rho_tau=ctx.rho_tau,
            rho_scale=ctx.rho_scale,
        )
        spike_credit = torch.zeros(
            (*grad_spikes.shape[:2], epsilon.shape[0], grad_spikes.shape[-1]),
            device=grad_spikes.device,
            dtype=grad_spikes.dtype,
        )
        for lag in range(min(grad_spikes.shape[-1], epsilon.shape[-1])):
            spike_credit[..., : grad_spikes.shape[-1] - lag] += (
                grad_spikes[..., lag:].unsqueeze(2)
                * epsilon[:, lag].view(1, 1, -1, 1)
            )
        delta = rho.unsqueeze(2) * spike_credit
        weight_value = weight.to(device=inputs.device, dtype=inputs.dtype)
        grad_weight = torch.einsum("boit,bit->oi", delta, activity) * ctx.dt
        error = torch.einsum("oi,boit->bit", weight_value, delta)
        if grad_membrane is not None:
            grad_weight = grad_weight + SLAYERLayer.weight_gradient(
                grad_membrane, activity, dt=ctx.dt
            )
            error = error + torch.einsum("oi,bot->bit", weight_value, grad_membrane)
        if grad_activity is not None:
            error = error + grad_activity
        grad_inputs = _future_correlation(error, epsilon) * ctx.dt
        grad_weight = grad_weight.to(device=weight.device, dtype=weight.dtype)
        grad_delays = None
        if ctx.learnable_delays:
            activity_derivative = _time_derivative(activity, ctx.dt)
            grad_delays = SLAYERLayer.delay_gradient(activity_derivative, error, dt=ctx.dt)
            grad_delays = grad_delays.to(device=delays.device, dtype=delays.dtype)
        return grad_inputs, grad_weight, grad_delays, None, None, None, None, None, None, None


class SLAYERLayer(nn.Module):
    component_id = "slayer"
    PRESETS = {
        "tiny": {"in_features": 2, "out_features": 2},
        "small": {"in_features": 16, "out_features": 8},
        "base": {"in_features": 64, "out_features": 32},
    }

    def __init__(
        self,
        in_features,
        out_features,
        threshold=10.0,
        tau_syn=10.0,
        tau_ref=1.0,
        scale_ref=2.0,
        rho_tau=1.0,
        rho_scale=1.0,
        dt=1.0,
        learnable_delays=False,
        initial_delays=None,
        synaptic_kernel=None,
        refractory_kernel=None,
    ):
        super().__init__()
        for name, value in (("in_features", in_features), ("out_features", out_features)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
            ("threshold", threshold),
            ("tau_syn", tau_syn),
            ("tau_ref", tau_ref),
            ("rho_tau", rho_tau),
            ("dt", dt),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if not isinstance(scale_ref, (int, float)) or isinstance(scale_ref, bool) or not math.isfinite(scale_ref) or scale_ref < 0:
            raise ValueError("scale_ref must be a finite non-negative number")
        if not isinstance(rho_scale, (int, float)) or isinstance(rho_scale, bool) or not math.isfinite(rho_scale) or rho_scale <= 0:
            raise ValueError("rho_scale must be a finite positive number")
        if not isinstance(learnable_delays, bool):
            invalid_value = ValueError("learnable_delays must be a boolean")
            raise invalid_value
        if initial_delays is not None:
            if not learnable_delays:
                raise ValueError("initial_delays requires learnable_delays=True")
            if not isinstance(initial_delays, (list, tuple)) or len(initial_delays) != in_features:
                raise ValueError("initial_delays must contain one value per input feature")
            if any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
                for value in initial_delays
            ):
                raise ValueError("initial_delays must contain finite non-negative values")
        epsilon = self._validate_kernel(synaptic_kernel, "synaptic_kernel")
        refractory = self._validate_kernel(refractory_kernel, "refractory_kernel")
        if refractory is not None and refractory[0] != 0.0:
            raise ValueError("refractory_kernel must start at zero")
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = float(threshold)
        self.tau_syn = float(tau_syn)
        self.tau_ref = float(tau_ref)
        self.scale_ref = float(scale_ref)
        self.rho_tau = float(rho_tau)
        self.rho_scale = float(rho_scale)
        self.dt = float(dt)
        self.learnable_delays = learnable_delays
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        if learnable_delays:
            values = torch.zeros(in_features) if initial_delays is None else torch.tensor(initial_delays, dtype=torch.float32)
            self.delays = nn.Parameter(values)
        else:
            self.register_parameter("delays", None)
        self.register_buffer(
            "synaptic_kernel",
            None if epsilon is None else torch.tensor(epsilon, dtype=torch.float32),
        )
        self.register_buffer(
            "refractory_kernel",
            None if refractory is None else torch.tensor(refractory, dtype=torch.float32),
        )

    @staticmethod
    def _validate_kernel(kernel, name):
        if kernel is None:
            return None
        if not isinstance(kernel, (list, tuple)) or not kernel:
            raise ValueError(f"{name} must be a non-empty sequence of finite numbers")
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in kernel
        ):
            raise ValueError(f"{name} must be a non-empty sequence of finite numbers")
        return [float(value) for value in kernel]

    @staticmethod
    def filter_spikes(spikes, kernel, dt=1.0):
        if not torch.is_tensor(spikes) or spikes.ndim != 3:
            raise ValueError("spikes must have shape [batch, features, time]")
        if not torch.is_tensor(kernel) or kernel.ndim not in (1, 2) or kernel.shape[-1] == 0:
            raise ValueError("kernel must be a non-empty one- or two-dimensional tensor")
        if not isinstance(dt, (int, float)) or isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be a finite positive number")
        return _causal_filter(spikes, kernel.to(device=spikes.device, dtype=spikes.dtype), float(dt))

    @staticmethod
    def van_rossum_loss(spikes, target, kernel, dt=1.0, reduction="mean"):
        if not torch.is_tensor(spikes) or not torch.is_tensor(target) or spikes.shape != target.shape or spikes.ndim != 3:
            raise ValueError("spikes and target must have matching [batch, features, time] shapes")
        filtered_spikes = SLAYERLayer.filter_spikes(spikes, kernel, dt)
        filtered_target = SLAYERLayer.filter_spikes(target, kernel, dt)
        per_example = 0.5 * dt * (filtered_spikes - filtered_target).square().sum(dim=(1, 2))
        if reduction == "none":
            return per_example
        if reduction == "sum":
            return per_example.sum()
        if reduction == "mean":
            return per_example.mean()
        raise ValueError("reduction must be 'none', 'sum', or 'mean'")

    @staticmethod
    def temporal_credit(error, kernel):
        if not torch.is_tensor(error) or error.ndim < 2:
            raise ValueError("error must have a feature dimension and a time dimension")
        if not torch.is_tensor(kernel) or kernel.ndim not in (1, 2) or kernel.shape[-1] == 0:
            raise ValueError("kernel must be a non-empty one- or two-dimensional tensor")
        return _future_correlation(
            error, kernel.to(device=error.device, dtype=error.dtype)
        )

    @staticmethod
    def weight_gradient(delta, activity, dt=1.0):
        if not torch.is_tensor(delta) or not torch.is_tensor(activity) or delta.ndim != 3 or activity.ndim != 3:
            raise ValueError("delta and activity must have shape [batch, features, time]")
        if delta.shape[0] != activity.shape[0] or delta.shape[-1] != activity.shape[-1]:
            raise ValueError("delta and activity must match in batch and time dimensions")
        if not isinstance(dt, (int, float)) or isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be a finite positive number")
        return torch.einsum("bot,bit->oi", delta, activity) * dt

    @staticmethod
    def surrogate_derivative(membrane, threshold, rho_tau=1.0, rho_scale=1.0):
        if not torch.is_tensor(membrane):
            raise ValueError("membrane must be a tensor")
        for name, value in (("threshold", threshold), ("rho_tau", rho_tau), ("rho_scale", rho_scale)):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        scale = rho_tau * threshold
        return (rho_scale / scale) * torch.exp(-torch.abs(membrane - threshold) / scale)

    @staticmethod
    def delay_gradient(activity_derivative, error, dt=1.0):
        if not torch.is_tensor(activity_derivative) or not torch.is_tensor(error) or activity_derivative.shape != error.shape or error.ndim != 3:
            raise ValueError("activity_derivative and error must have matching [batch, features, time] shapes")
        if not isinstance(dt, (int, float)) or isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be a finite positive number")
        return -dt * (activity_derivative * error).sum(dim=(0, 2))

    def _kernel_values(self, inputs, kernel, tau, refractory=False):
        if kernel is not None:
            return kernel.to(device=inputs.device, dtype=inputs.dtype)
        times = torch.arange(inputs.shape[-1], device=inputs.device, dtype=inputs.dtype) * self.dt
        scaled = times / tau
        values = scaled * torch.exp(1.0 - scaled)
        if refractory:
            values = -self.scale_ref * self.threshold * values
        return values

    def forward(self, inputs, return_state=False):
        if not torch.is_tensor(inputs) or inputs.ndim != 3:
            raise ValueError("inputs must have shape [batch, in_features, time]")
        if inputs.shape[1] != self.in_features:
            raise ValueError(f"input feature dimension must be {self.in_features}")
        if not torch.is_floating_point(inputs):
            raise ValueError("inputs must have a floating-point dtype")
        if inputs.shape[-1] < 1:
            raise ValueError("time dimension must be positive")
        if not isinstance(return_state, bool):
            raise TypeError("return_state must be a boolean")
        epsilon = self._kernel_values(inputs, self.synaptic_kernel, self.tau_syn)
        refractory = self._kernel_values(inputs, self.refractory_kernel, self.tau_ref, True)
        delays = (
            self.delays
            if self.delays is not None
            else inputs.new_zeros(self.in_features)
        )
        spikes, activity, membrane = _SLAYERFunction.apply(
            inputs,
            self.weight,
            delays,
            epsilon,
            refractory,
            self.dt,
            self.threshold,
            self.rho_tau,
            self.rho_scale,
            self.learnable_delays,
        )
        if return_state:
            return spikes, activity, membrane
        return spikes
