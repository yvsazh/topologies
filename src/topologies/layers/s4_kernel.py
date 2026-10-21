import torch
from torch import nn
import torch.nn.functional as F


def hippo_legs(size, dtype=torch.float32, device=None):
    if size < 1:
        raise ValueError("size must be positive")
    index = torch.arange(size, dtype=dtype, device=device)
    row = index[:, None]
    col = index[None, :]
    return -torch.tril(torch.sqrt(2 * row + 1) * torch.sqrt(2 * col + 1), diagonal=-1) - torch.diag(torch.arange(1, size + 1, dtype=dtype, device=device))


def cauchy_kernel(z, poles):
    return 1 / (z[:, None] - poles[None, :])


class S4Kernel(nn.Module):
    component_id = "s4-kernel"
    PRESETS = {"tiny": {"d_model": 2, "state_size": 4, "dropout": 0.0}}

    def __init__(self, d_model=8, state_size=8, dropout=0.0):
        super().__init__()
        if d_model < 1 or state_size < 1:
            raise ValueError("d_model and state_size must be positive")
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        legs = hippo_legs(state_size, dtype=torch.float64)
        index = torch.arange(state_size, dtype=torch.float64)
        hippo_p = torch.sqrt(index + 0.5)
        normal = legs + torch.outer(hippo_p, hippo_p)
        eigenvalues, eigenvectors = torch.linalg.eig(normal)
        transformed_p = eigenvectors.mH @ hippo_p.to(torch.complex128)
        self.d_model = d_model
        self.state_size = state_size
        self.register_buffer("eigenvalues", eigenvalues.to(torch.complex64))
        self.register_buffer("rank_left", transformed_p.to(torch.complex64))
        self.register_buffer("rank_right", transformed_p.to(torch.complex64))
        self.log_dt = nn.Parameter(torch.full((d_model,), -5.0))
        self.B = nn.Parameter(torch.ones(d_model, state_size, dtype=torch.complex64) / state_size ** 0.5)
        self.C = nn.Parameter(torch.ones(d_model, state_size, dtype=torch.complex64) / state_size ** 0.5)
        self.D = nn.Parameter(torch.ones(d_model))

    def discrete_parameters(self):
        dt = self.log_dt.exp().to(self.B.real.dtype)
        poles = 1 + dt[:, None] * self.eigenvalues[None, :]
        left = dt[:, None] * self.rank_left[None, :]
        right = self.rank_right.unsqueeze(0).expand(self.d_model, -1)
        input_factor = dt[:, None] * self.B
        return poles, left, right, input_factor, dt

    def recurrent(self, inputs):
        poles, left, right, input_factor, _ = self.discrete_parameters()
        state = torch.zeros(inputs.shape[0], self.d_model, self.state_size, dtype=self.B.dtype, device=inputs.device)
        outputs = []
        for item in inputs.unbind(1):
            projection = torch.einsum("dn,bdn->bd", right.conj(), state)
            state = poles.unsqueeze(0) * state - left.unsqueeze(0) * projection.unsqueeze(-1)
            state = state + input_factor.unsqueeze(0) * item.unsqueeze(-1)
            result = torch.einsum("dn,bdn->bd", self.C, state) + self.D * item
            outputs.append(result.real)
        return torch.stack(outputs, dim=1)

    def convolution(self, inputs):
        length = inputs.shape[1]
        poles, left, right, input_factor, _ = self.discrete_parameters()
        fft_length = max(4096, 16 * length)
        angles = 2 * torch.pi * torch.arange(fft_length, dtype=self.B.real.dtype, device=inputs.device) / fft_length
        z = torch.polar(torch.ones_like(angles), angles)
        z = z.to(self.B.dtype)
        values = []
        for channel in range(self.d_model):
            cauchy = cauchy_kernel(z, poles[channel])
            diagonal_inverse_b = cauchy * input_factor[channel][None, :]
            inverse_left = cauchy * left[channel][None, :]
            denominator = 1 + (inverse_left * right[channel].conj()[None, :]).sum(dim=-1)
            correction = (diagonal_inverse_b * right[channel].conj()[None, :]).sum(dim=-1) / denominator
            response = z * (self.C[channel][None, :] * (diagonal_inverse_b - inverse_left * correction[:, None])).sum(dim=-1)
            values.append(response)
        spectrum = torch.stack(values, dim=0)
        kernel = torch.fft.ifft(spectrum, dim=-1).real[..., :length]
        output = F.conv1d(inputs.transpose(1, 2), kernel.flip(-1).unsqueeze(1), padding=length - 1, groups=self.d_model)
        return output[..., :length].transpose(1, 2) + inputs * self.D

    def forward(self, inputs, mode="convolution"):
        if inputs.ndim != 3 or inputs.shape[-1] != self.d_model:
            raise ValueError("inputs must have shape (batch, length, d_model)")
        if mode == "recurrent":
            return self.recurrent(inputs)
        if mode != "convolution":
            raise ValueError("mode must be convolution or recurrent")
        return self.convolution(inputs)
