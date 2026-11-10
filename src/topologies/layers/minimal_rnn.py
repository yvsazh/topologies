import torch
from torch import nn


class MinimalRNN(nn.Module):
    component_id = "minimal-rnn"

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.W_x = nn.Parameter(torch.empty(hidden_dim, input_dim))
        self.b_z = nn.Parameter(torch.zeros(hidden_dim))
        self.U_h = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.U_z = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.b_u = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.xavier_uniform_(self.W_x)
        nn.init.xavier_uniform_(self.U_h)
        nn.init.xavier_uniform_(self.U_z)

    def forward(self, x, initial_state=None):
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError("input must have shape [batch, sequence, input_dim]")
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"input feature dimension must be {self.input_dim}, got {x.shape[-1]}"
            )
        batch_size, sequence_length, _ = x.shape
        if initial_state is None:
            state = x.new_zeros((batch_size, self.hidden_dim))
        else:
            if not isinstance(initial_state, torch.Tensor) or initial_state.shape != (
                batch_size,
                self.hidden_dim,
            ):
                raise ValueError(
                    "initial_state must have shape [batch, hidden_dim] matching the input"
                )
            state = initial_state
        if sequence_length == 0:
            return x.new_empty((batch_size, 0, self.hidden_dim))
        states = []
        for time_index in range(sequence_length):
            encoded = torch.tanh(
                torch.nn.functional.linear(x[:, time_index, :], self.W_x, self.b_z)
            )
            update = torch.sigmoid(
                torch.nn.functional.linear(state, self.U_h)
                + torch.nn.functional.linear(encoded, self.U_z, self.b_u)
            )
            state = update * state + (1.0 - update) * encoded
            states.append(state)
        return torch.stack(states, dim=1)
