import torch
from torch import nn
from torch.nn import functional as F


class QRNN(nn.Module):
    architecture_id = "qrnn"
    PRESETS = {
        "tiny": {
            "input_size": 8,
            "hidden_size": 8,
            "kernel_size": 2,
            "num_layers": 1,
            "pooling": "f",
        },
        "small": {
            "input_size": 64,
            "hidden_size": 64,
            "kernel_size": 2,
            "num_layers": 2,
            "pooling": "f",
        },
    }

    def __init__(
        self,
        input_size,
        hidden_size,
        kernel_size=2,
        num_layers=2,
        pooling="f",
    ):
        super().__init__()
        dimensions = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "kernel_size": kernel_size,
            "num_layers": num_layers,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if pooling not in ("f", "fo", "ifo"):
            raise ValueError("pooling must be one of 'f', 'fo', or 'ifo'")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.pooling = pooling
        layer_input_sizes = [input_size] + [hidden_size] * (num_layers - 1)
        self.candidate_convs = nn.ModuleList(
            nn.Conv1d(size, hidden_size, kernel_size, bias=False)
            for size in layer_input_sizes
        )
        self.forget_convs = nn.ModuleList(
            nn.Conv1d(size, hidden_size, kernel_size, bias=False)
            for size in layer_input_sizes
        )
        self.input_convs = nn.ModuleList(
            nn.Conv1d(size, hidden_size, kernel_size, bias=False)
            for size in layer_input_sizes
        ) if pooling == "ifo" else nn.ModuleList()
        self.output_convs = nn.ModuleList(
            nn.Conv1d(size, hidden_size, kernel_size, bias=False)
            for size in layer_input_sizes
        ) if pooling != "f" else nn.ModuleList()

    def _convolve(self, layer, inputs):
        padded = F.pad(inputs.transpose(1, 2), (self.kernel_size - 1, 0))
        return layer(padded).transpose(1, 2)

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError("input must be a tensor with shape (batch, time, features)")
        if x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError("input batch and time dimensions must be nonempty")
        if x.shape[2] != self.input_size:
            raise ValueError(f"input feature width must be {self.input_size}")

        layer_inputs = x
        for index in range(self.num_layers):
            candidate = torch.tanh(self._convolve(self.candidate_convs[index], layer_inputs))
            forget = torch.sigmoid(self._convolve(self.forget_convs[index], layer_inputs))
            input_gate = None
            output_gate = None
            if self.pooling == "ifo":
                input_gate = torch.sigmoid(self._convolve(self.input_convs[index], layer_inputs))
            if self.pooling != "f":
                output_gate = torch.sigmoid(self._convolve(self.output_convs[index], layer_inputs))

            state = candidate.new_zeros(candidate.shape[0], self.hidden_size)
            outputs = []
            for step in range(candidate.shape[1]):
                if self.pooling == "ifo":
                    state = forget[:, step] * state + input_gate[:, step] * candidate[:, step]
                else:
                    state = forget[:, step] * state + (1 - forget[:, step]) * candidate[:, step]
                outputs.append(state if output_gate is None else output_gate[:, step] * state)
            layer_inputs = torch.stack(outputs, dim=1)
        return layer_inputs
