import torch
from torch import nn


class GRU(nn.Module):
    architecture_id = "gru"
    PRESETS = {
        "tiny": {"input_size": 93, "hidden_size": 46},
        "small": {"input_size": 20, "hidden_size": 227},
    }

    def __init__(self, input_size=93, hidden_size=46):
        super().__init__()
        if type(input_size) is not int or input_size <= 0:
            raise ValueError("input_size must be a positive integer")
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_z = nn.Parameter(torch.empty(hidden_size, input_size))
        self.U_z = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_r = nn.Parameter(torch.empty(hidden_size, input_size))
        self.U_r = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_h = nn.Parameter(torch.empty(hidden_size, input_size))
        self.U_h = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        for weight in (self.W_z, self.U_z, self.W_r, self.U_r, self.W_h, self.U_h):
            nn.init.kaiming_uniform_(weight, a=5 ** 0.5)

    def _step(self, x, previous):
        update = torch.sigmoid(x @ self.W_z.T + previous @ self.U_z.T)
        reset = torch.sigmoid(x @ self.W_r.T + previous @ self.U_r.T)
        candidate = torch.tanh(x @ self.W_h.T + (reset * previous) @ self.U_h.T)
        next_state = (1 - update) * previous + update * candidate
        return next_state, update, reset, candidate

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError("x must be a tensor with shape [B,T,input_size]")
        if x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError("x batch and sequence dimensions must be nonempty")
        if x.shape[2] != self.input_size:
            raise ValueError(f"x final dimension must equal input_size ({self.input_size})")
        if not x.is_floating_point():
            raise ValueError("x must have a floating-point dtype")
        if x.dtype != self.W_z.dtype:
            raise ValueError("x and model parameters must have the same dtype")
        if x.device != self.W_z.device:
            raise ValueError("x and model parameters must be on the same device")

        hidden = x.new_zeros(x.shape[0], self.hidden_size)
        states = []
        for step_input in x.unbind(dim=1):
            hidden, _, _, _ = self._step(step_input, hidden)
            states.append(hidden)
        return torch.stack(states, dim=1)
