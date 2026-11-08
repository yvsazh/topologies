import math

import torch
from torch import nn
from torch.nn import functional as F


class Mamba2(nn.Module):
    component_id = "mamba2"
    PRESETS = {
        "tiny": {"d_model": 4, "d_state": 2, "d_conv": 2, "expand": 1, "headdim": 2, "ngroups": 1},
        "small": {"d_model": 16, "d_state": 8, "d_conv": 4, "expand": 2, "headdim": 4, "ngroups": 1},
        "base": {"d_model": 64, "d_state": 64, "d_conv": 4, "expand": 2, "headdim": 64, "ngroups": 1},
    }

    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        expand=2,
        headdim=64,
        ngroups=1,
        A_init_range=(1.0, 16.0),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        rmsnorm=True,
        norm_before_gate=False,
        norm_eps=1e-5,
        bias=False,
        conv_bias=True,
    ):
        super().__init__()
        dimensions = {
            "d_model": d_model,
            "d_state": d_state,
            "d_conv": d_conv,
            "expand": expand,
            "headdim": headdim,
            "ngroups": ngroups,
        }
        for name, value in dimensions.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(A_init_range, (tuple, list)) or len(A_init_range) != 2:
            raise ValueError("A_init_range must contain two positive finite values")
        a_min, a_max = A_init_range
        if (
            not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) for value in (a_min, a_max))
            or a_min <= 0
            or a_max < a_min
        ):
            raise ValueError("A_init_range must contain ordered positive finite values")
        rates = {"dt_min": dt_min, "dt_max": dt_max, "dt_init_floor": dt_init_floor}
        for name, value in rates.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        if dt_max < dt_min:
            raise ValueError("dt_max must be greater than or equal to dt_min")
        if not isinstance(norm_eps, (int, float)) or isinstance(norm_eps, bool) or not math.isfinite(norm_eps) or norm_eps <= 0:
            raise ValueError("norm_eps must be a positive finite number")
        flags = {
            "rmsnorm": rmsnorm,
            "norm_before_gate": norm_before_gate,
            "bias": bias,
            "conv_bias": conv_bias,
        }
        if any(value is not True and value is not False for value in flags.values()):
            raise ValueError(f"boolean constructor options must be bool values: {tuple(flags)}")

        inner_dim = expand * d_model
        if inner_dim % headdim != 0:
            raise ValueError("headdim must divide expand * d_model")
        nheads = inner_dim // headdim
        if ngroups > nheads or nheads % ngroups != 0:
            raise ValueError("ngroups must divide the number of heads")

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.headdim = headdim
        self.ngroups = ngroups
        self.inner_dim = inner_dim
        self.nheads = nheads
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.norm_eps = float(norm_eps)
        self.in_proj = nn.Linear(d_model, 2 * inner_dim + 2 * ngroups * d_state + nheads, bias=bias)
        conv_dim = inner_dim + 2 * ngroups * d_state
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, d_conv, groups=conv_dim, bias=conv_bias)
        self.out_proj = nn.Linear(inner_dim, d_model, bias=bias)

        with torch.no_grad():
            a_values = torch.empty(nheads).uniform_(float(a_min), float(a_max))
            self.A_log = nn.Parameter(a_values.log())
            dt_values = torch.exp(
                torch.empty(nheads).uniform_(math.log(float(dt_min)), math.log(float(dt_max)))
            ).clamp_min(float(dt_init_floor))
            inverse_softplus = dt_values + torch.log(-torch.expm1(-dt_values))
            self.dt_bias = nn.Parameter(inverse_softplus)
            self.D = nn.Parameter(torch.ones(nheads))
            self.norm_weight = nn.Parameter(torch.ones(inner_dim)) if rmsnorm else None

    def _rms_norm(self, value):
        normalized = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + self.norm_eps)
        return normalized * self.norm_weight

    def _gated_norm(self, value, gate):
        if self.rmsnorm and self.norm_before_gate:
            return self._rms_norm(value) * F.silu(gate)
        gated = value * F.silu(gate)
        return self._rms_norm(gated) if self.rmsnorm else gated

    def _ssd_recurrence(self, x, dt, B, C):
        batch, length, _, _ = x.shape
        state = x.new_zeros((batch, self.nheads, self.headdim, self.d_state))
        A = -self.A_log.exp().view(1, self.nheads, 1, 1)
        outputs = []
        for index in range(length):
            dt_t = dt[:, index].unsqueeze(-1).unsqueeze(-1)
            state = (
                torch.exp(dt_t * A) * state
                + dt_t * x[:, index].unsqueeze(-1) * B[:, index].unsqueeze(2)
            )
            readout = (state * C[:, index].unsqueeze(2)).sum(dim=-1)
            outputs.append(readout + self.D.view(1, self.nheads, 1) * x[:, index])
        return torch.stack(outputs, dim=1), state

    def forward(self, inputs):
        if not torch.is_tensor(inputs) or inputs.ndim != 3:
            raise ValueError("input must have rank 3 with shape [batch, time, d_model]")
        if inputs.shape[-1] != self.d_model:
            raise ValueError(f"input last dimension must equal d_model ({self.d_model})")
        if inputs.shape[1] <= 0:
            raise ValueError("input sequence length must be positive")

        batch, length, _ = inputs.shape
        group_state = self.ngroups * self.d_state
        sizes = (self.inner_dim, self.inner_dim, group_state, group_state, self.nheads)
        z, x, B, C, dt_logits = self.in_proj(inputs).split(sizes, dim=-1)
        conv_input = torch.cat((x, B, C), dim=-1).transpose(1, 2)
        convolved = self.conv1d(F.pad(conv_input, (self.d_conv - 1, 0))).transpose(1, 2)
        convolved = F.silu(convolved)
        x, B, C = convolved.split((self.inner_dim, group_state, group_state), dim=-1)
        x = x.reshape(batch, length, self.nheads, self.headdim)
        B = B.reshape(batch, length, self.ngroups, self.d_state)
        C = C.reshape(batch, length, self.ngroups, self.d_state)
        heads_per_group = self.nheads // self.ngroups
        B = B.repeat_interleave(heads_per_group, dim=2)
        C = C.repeat_interleave(heads_per_group, dim=2)
        dt = F.softplus(dt_logits + self.dt_bias)
        output, _ = self._ssd_recurrence(x, dt, B, C)
        output = output.reshape(batch, length, self.inner_dim)
        output = self._gated_norm(output, z)
        return self.out_proj(output)
