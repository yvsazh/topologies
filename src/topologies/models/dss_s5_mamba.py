import torch
from torch import nn
import torch.nn.functional as F


class DSSS5Mamba(nn.Module):
    architecture_id = "dss-s5-mamba"
    component_id = "dss-s5-mamba"
    PRESETS = {
        "tiny": {"vocab_size": 32, "d_model": 16, "state_size": 8, "num_layers": 1, "mode": "dss"},
        "base": {"vocab_size": 256, "d_model": 64, "state_size": 16, "num_layers": 4, "mode": "mamba"},
    }
    BUDGET_KNOBS = {"d_model": range(8, 129, 8)}

    def __init__(self, vocab_size=32, d_model=16, state_size=8, num_layers=1, mode="dss", dropout=0.0):
        super().__init__()
        if vocab_size < 2 or d_model < 1 or state_size < 1 or num_layers < 1:
            raise ValueError("vocabulary, model, state, and layer sizes must be positive")
        if mode not in {"dss", "s5", "mamba"}:
            raise ValueError("mode must be dss, s5, or mamba")
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.state_size = state_size
        self.num_layers = num_layers
        self.mode = mode
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([SSMBlock(d_model, state_size, mode) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens):
        if tokens.ndim != 2 or tokens.dtype not in (torch.int32, torch.int64):
            raise ValueError("tokens must be an integer tensor with shape (batch, length)")
        hidden = self.embedding(tokens)
        for layer in self.layers:
            hidden = hidden + layer(hidden)
        return self.output(self.norm(hidden))


class SSMBlock(nn.Module):
    def __init__(self, d_model, state_size, mode):
        super().__init__()
        self.d_model = d_model
        self.state_size = state_size
        self.mode = mode
        self.input_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        if mode == "dss":
            self.log_decay = nn.Parameter(torch.full((d_model, state_size), -0.2))
            self.frequency = nn.Parameter(torch.ones(d_model, state_size))
            self.B = nn.Parameter(torch.ones(d_model, state_size) / state_size**0.5)
            self.C = nn.Parameter(torch.ones(d_model, state_size) / state_size**0.5)
            self.D = nn.Parameter(torch.ones(d_model))
        elif mode == "s5":
            self.log_decay = nn.Parameter(torch.full((state_size,), -0.2))
            self.B = nn.Parameter(torch.randn(state_size, d_model) / d_model**0.5)
            self.C = nn.Parameter(torch.randn(d_model, state_size) / state_size**0.5)
        else:
            self.log_decay = nn.Parameter(torch.full((state_size,), -0.2))
            self.delta_projection = nn.Linear(d_model, d_model * state_size)
            self.B_projection = nn.Linear(d_model, d_model * state_size)
            self.C_projection = nn.Linear(d_model, d_model * state_size)

    def _input(self, inputs):
        return torch.tanh(self.input_projection(inputs))

    def dss_kernel(self, length):
        positions = torch.arange(length, device=self.log_decay.device, dtype=self.log_decay.dtype)
        poles = torch.exp(self.log_decay + 1j * self.frequency)
        powers = poles.unsqueeze(-1) ** positions
        impulse = (powers * (self.C * self.B).unsqueeze(-1).to(powers.dtype)).sum(dim=1).real
        return impulse.transpose(0, 1)

    def dss_recurrent(self, inputs):
        values = self._input(inputs)
        poles = torch.exp(self.log_decay + 1j * self.frequency)
        state = torch.zeros(values.shape[0], self.d_model, self.state_size, dtype=torch.complex64, device=values.device)
        output = []
        for value in values.unbind(1):
            state = poles * state + value.unsqueeze(-1) * self.B.to(torch.complex64)
            output.append((state * self.C.to(torch.complex64)).sum(-1).real + value * self.D)
        return torch.stack(output, dim=1)

    def dss_convolution(self, inputs):
        values = self._input(inputs)
        kernel = self.dss_kernel(values.shape[1]).transpose(0, 1).unsqueeze(1)
        convolved = F.conv1d(values.transpose(1, 2), kernel.flip(-1), padding=values.shape[1] - 1, groups=self.d_model)
        return convolved[..., :values.shape[1]].transpose(1, 2) + values * self.D

    def s5_step(self, value, state):
        next_state = torch.exp(self.log_decay).view(1, -1) * state + value @ self.B.transpose(0, 1)
        return next_state @ self.C.transpose(0, 1), next_state

    def s5_scan(self, inputs):
        values = self._input(inputs)
        a = torch.exp(self.log_decay).view(1, 1, -1).expand(values.shape[0], values.shape[1], -1)
        b = values @ self.B.transpose(0, 1)
        _, states = self._affine_prefix_scan(a, b)
        return states @ self.C.transpose(0, 1)

    @staticmethod
    def _affine_prefix_scan(a, b):
        offset = 1
        while offset < b.shape[1]:
            old_a, old_b = a, b
            a = torch.cat((old_a[:, :offset], old_a[:, offset:] * old_a[:, :-offset]), dim=1)
            b = torch.cat((old_b[:, :offset], old_b[:, offset:] + old_a[:, offset:] * old_b[:, :-offset]), dim=1)
            offset *= 2
        return a, b

    def s5_recurrent(self, inputs):
        values = self._input(inputs)
        state = values.new_zeros(values.shape[0], self.state_size)
        outputs = []
        for value in values.unbind(1):
            output, state = self.s5_step(value, state)
            outputs.append(output)
        return torch.stack(outputs, dim=1)

    def mamba_parameters(self, inputs):
        batch, length, _ = inputs.shape
        shape = (batch, length, self.d_model, self.state_size)
        delta = F.softplus(self.delta_projection(inputs).reshape(shape))
        b = torch.tanh(self.B_projection(inputs).reshape(shape))
        c = torch.tanh(self.C_projection(inputs).reshape(shape))
        decay = torch.exp(-delta * F.softplus(self.log_decay).view(1, 1, 1, -1))
        return decay, b, c

    def mamba_recurrent(self, inputs):
        values = self._input(inputs)
        decay, b, c = self.mamba_parameters(values)
        state = values.new_zeros(values.shape[0], self.d_model, self.state_size)
        outputs = []
        for index, value in enumerate(values.unbind(1)):
            state = decay[:, index] * state + b[:, index] * value.unsqueeze(-1)
            outputs.append((c[:, index] * state).sum(-1))
        return torch.stack(outputs, dim=1)

    def mamba_scan(self, inputs):
        values = self._input(inputs)
        decay, b, c = self.mamba_parameters(values)
        _, states = self._affine_prefix_scan(decay, b * values.unsqueeze(-1))
        return (c * states).sum(-1)

    def forward(self, inputs):
        if inputs.ndim != 3 or inputs.shape[-1] != self.d_model:
            raise ValueError("inputs must have shape (batch, length, d_model)")
        if self.mode == "dss":
            result = self.dss_convolution(inputs)
        elif self.mode == "s5":
            result = self.s5_scan(inputs)
        else:
            result = self.mamba_scan(inputs)
        return self.output_projection(result)
