import torch
from torch import nn


class MAF(nn.Module):
    architecture_id = "maf"
    def __init__(self, input_dim=4, hidden_dim=32):
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        self.input_dim = input_dim
        self.shift_nets = nn.ModuleList()
        self.scale_nets = nn.ModuleList()
        for index in range(input_dim):
            self.shift_nets.append(nn.Sequential(nn.Linear(index, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)) if index else nn.Identity())
            self.scale_nets.append(nn.Sequential(nn.Linear(index, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)) if index else nn.Identity())

    def _parameters_for(self, index, prefix):
        if index == 0:
            zero = prefix.new_zeros(prefix.shape[:-1] + (1,))
            return zero, zero
        return self.shift_nets[index](prefix), self.scale_nets[index](prefix).tanh()

    def forward(self, inputs):
        if inputs.shape[-1] != self.input_dim:
            raise ValueError(f"expected last dimension {self.input_dim}")
        values = []
        for index in range(self.input_dim):
            prefix = inputs[..., :index]
            shift, log_scale = self._parameters_for(index, prefix)
            values.append((inputs[..., index:index + 1] - shift) * torch.exp(-log_scale))
        return torch.cat(values, dim=-1)

    def log_prob(self, inputs):
        if inputs.shape[-1] != self.input_dim:
            raise ValueError(f"expected last dimension {self.input_dim}")
        transformed = self.forward(inputs)
        log_scales = []
        for index in range(self.input_dim):
            prefix = inputs[..., :index]
            _, log_scale = self._parameters_for(index, prefix)
            log_scales.append(log_scale)
        log_det = -torch.cat(log_scales, dim=-1).sum(dim=-1)
        log_base = -0.5 * (transformed.square() + torch.log(inputs.new_tensor(2.0 * torch.pi))).sum(dim=-1)
        return log_base + log_det

    def inverse(self, outputs):
        if outputs.shape[-1] != self.input_dim:
            raise ValueError(f"expected last dimension {self.input_dim}")
        values = []
        for index in range(self.input_dim):
            prefix = torch.cat(values, dim=-1) if values else outputs[..., :0]
            shift, log_scale = self._parameters_for(index, prefix)
            values.append(outputs[..., index:index + 1] * torch.exp(log_scale) + shift)
        return torch.cat(values, dim=-1)
