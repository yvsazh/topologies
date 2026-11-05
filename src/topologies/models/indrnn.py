import torch
from torch import nn


class IndRNN(nn.Module):
    architecture_id = "indrnn"
    PRESETS = {
        "tiny": {"input_size": 2, "hidden_size": 4, "num_layers": 1},
        "small": {"input_size": 16, "hidden_size": 64, "num_layers": 2},
    }

    def __init__(self, input_size, hidden_size, num_layers=1, activation="relu", bias=True,
                 batch_first=True):
        super().__init__()
        dimensions = (input_size, hidden_size, num_layers)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
               for value in dimensions):
            raise ValueError("input_size, hidden_size, and num_layers must be positive integers")
        if not isinstance(activation, str) or activation not in ("relu", "tanh"):
            raise ValueError("activation must be 'relu' or 'tanh'")
        if type(bias) is not bool:
            raise ValueError("bias must be a boolean")
        if type(batch_first) is not bool:
            raise ValueError("batch_first must be a boolean")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation
        self.batch_first = batch_first
        self.input_maps = nn.ModuleList(
            nn.Linear(input_size if index == 0 else hidden_size, hidden_size, bias=bias)
            for index in range(num_layers)
        )
        self.recurrent_weights = nn.ParameterList(
            nn.Parameter(torch.ones(hidden_size)) for _ in range(num_layers)
        )

    def forward(self, x, initial_state=None, return_state=False):
        if not isinstance(x, torch.Tensor) or x.ndim != 3 or not x.is_floating_point():
            raise ValueError("x must be a floating-point rank-3 tensor")
        values = x if self.batch_first else x.transpose(0, 1)
        batch_size, sequence_length, input_size = values.shape
        if batch_size < 1 or sequence_length < 1:
            raise ValueError("x must have nonempty batch and sequence dimensions")
        if input_size != self.input_size:
            raise ValueError(f"x must have input feature width {self.input_size}")
        if type(return_state) is not bool:
            raise ValueError("return_state must be a boolean")
        if initial_state is None:
            states = [values.new_zeros(batch_size, self.hidden_size)
                      for _ in range(self.num_layers)]
        else:
            if not isinstance(initial_state, torch.Tensor) or not initial_state.is_floating_point():
                raise ValueError("initial_state must be a floating-point tensor")
            if initial_state.shape == (batch_size, self.hidden_size) and self.num_layers == 1:
                initial_state = initial_state.unsqueeze(0)
            expected_state_shape = (self.num_layers, batch_size, self.hidden_size)
            if tuple(initial_state.shape) != expected_state_shape:
                raise ValueError(f"initial_state must have shape {expected_state_shape}")
            initial_state = initial_state.to(device=values.device, dtype=values.dtype)
            states = [initial_state[index] for index in range(self.num_layers)]
        outputs = values
        final_states = []
        for layer, (input_map, recurrent_weight) in enumerate(
                zip(self.input_maps, self.recurrent_weights)):
            projected = input_map(outputs)
            state = states[layer]
            sequence = []
            for timestep in range(sequence_length):
                value = projected[:, timestep] + recurrent_weight * state
                state = torch.relu(value) if self.activation == "relu" else torch.tanh(value)
                sequence.append(state)
            outputs = torch.stack(sequence, dim=1)
            final_states.append(state)
        if not self.batch_first:
            outputs = outputs.transpose(0, 1)
        final_state = torch.stack(final_states, dim=0)
        return (outputs, final_state) if return_state else outputs
