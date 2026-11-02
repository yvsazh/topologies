import math

import torch
from torch import nn


def _is_finite_number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _invalid_argument(message):
    return ValueError(message)


class _BoxcarSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane, half_width):
        ctx.half_width = half_width
        ctx.save_for_backward(membrane)
        return (membrane >= 0).to(membrane.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (membrane,) = ctx.saved_tensors
        derivative = (membrane.abs() <= ctx.half_width).to(grad_output.dtype)
        return grad_output * derivative, None


class Decolle(nn.Module):
    component_id = "decolle"

    def __init__(
        self,
        in_features,
        hidden_features,
        target_features,
        alpha_mem=0.9,
        beta_syn=0.85,
        gamma_ref=0.65,
        rho=0.0,
        surrogate_half_width=0.5,
        readout_scale=0.5,
        lambda1=1.0,
        lambda2=0.005,
        seed=0,
    ):
        super().__init__()
        for name, value in (
            ("in_features", in_features),
            ("hidden_features", hidden_features),
            ("target_features", target_features),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
            ("alpha_mem", alpha_mem),
            ("beta_syn", beta_syn),
            ("gamma_ref", gamma_ref),
        ):
            if (
                not _is_finite_number(value)
                or value < 0
                or value >= 1
            ):
                raise ValueError(f"{name} must be finite and in [0, 1)")
        for name, value in (
            ("rho", rho),
            ("readout_scale", readout_scale),
            ("lambda1", lambda1),
            ("lambda2", lambda2),
        ):
            if (
                not _is_finite_number(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be a finite nonnegative number")
        if (
            not _is_finite_number(surrogate_half_width)
            or surrogate_half_width <= 0
        ):
            raise _invalid_argument("surrogate_half_width must be a finite positive number")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise _invalid_argument("seed must be an integer")

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.target_features = target_features
        self.alpha_mem = float(alpha_mem)
        self.beta_syn = float(beta_syn)
        self.gamma_ref = float(gamma_ref)
        self.rho = float(rho)
        self.surrogate_half_width = float(surrogate_half_width)
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.weight = nn.Parameter(torch.zeros(hidden_features, in_features))
        self.bias = nn.Parameter(torch.zeros(hidden_features))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        readout = torch.empty(target_features, hidden_features).uniform_(
            -float(readout_scale), float(readout_scale), generator=generator
        )
        self.register_buffer("readout", readout)
        self.register_buffer("p_state", None)
        self.register_buffer("q_state", None)
        self.register_buffer("r_state", None)
        self.aux_loss = torch.zeros(())
        self.voltage_loss = torch.zeros(())
        self.activity_loss = torch.zeros(())
        self.spikes = None
        self.membrane = None
        self.readout_output = None

    def reset_state(self):
        if self.p_state is not None:
            self.p_state = torch.zeros_like(self.p_state)
            self.q_state = torch.zeros_like(self.q_state)
            self.r_state = torch.zeros_like(self.r_state)
        self.aux_loss = self.weight.new_zeros(())
        self.voltage_loss = self.weight.new_zeros(())
        self.activity_loss = self.weight.new_zeros(())

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise _invalid_argument("x must be a tensor with shape [batch, in_features]")
        if x.ndim != 2 or x.shape[0] <= 0 or x.shape[1] != self.in_features:
            raise ValueError(f"x must have shape [batch, {self.in_features}] with a positive batch size")
        if not x.is_floating_point():
            raise ValueError("x must have a floating point dtype")
        if x.device != self.weight.device or x.dtype != self.weight.dtype:
            raise ValueError("x must match the layer parameter device and dtype")

        expected_shape = (x.shape[0], self.in_features)
        if (
            self.p_state is None
            or tuple(self.p_state.shape) != expected_shape
            or self.p_state.device != x.device
            or self.p_state.dtype != x.dtype
        ):
            p = x.new_zeros(expected_shape)
            q = x.new_zeros(expected_shape)
            r = x.new_zeros((x.shape[0], self.hidden_features))
        else:
            p = self.p_state
            q = self.q_state
            r = self.r_state

        membrane = p @ self.weight.T - r * self.rho + self.bias
        spikes = _BoxcarSpike.apply(membrane, self.surrogate_half_width)
        readout = spikes @ self.readout.T
        self.spikes = spikes
        self.membrane = membrane
        self.readout_output = readout
        self.voltage_loss = self.lambda1 * torch.relu(membrane + 0.01).mean()
        self.activity_loss = self.lambda2 * torch.relu(0.1 - membrane.mean(dim=-1)).mean()
        self.aux_loss = self.voltage_loss + self.activity_loss

        self.p_state = (self.alpha_mem * p + (1.0 - self.alpha_mem) * q).detach()
        self.q_state = (self.beta_syn * q + (1.0 - self.beta_syn) * x).detach()
        self.r_state = (self.gamma_ref * r + (1.0 - self.gamma_ref) * spikes).detach()

        return readout
