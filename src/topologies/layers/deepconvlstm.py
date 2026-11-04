import math

import torch
from torch import nn
from torch.nn import functional as F


class DeepConvLSTM(nn.Module):
    component_id = "deepconvlstm"
    PRESETS = {
        "tiny": {"conv_channels": 4, "kernel_size": 5, "hidden_size": 8, "dropout": 0.5},
        "small": {"conv_channels": 16, "kernel_size": 5, "hidden_size": 32, "dropout": 0.5},
        "base": {"conv_channels": 64, "kernel_size": 5, "hidden_size": 128, "dropout": 0.5},
    }

    def __init__(self, input_channels, num_classes, conv_channels=64, kernel_size=5, hidden_size=128, dropout=0.5):
        super().__init__()
        dimensions = {
            "input_channels": input_channels,
            "num_classes": num_classes,
            "conv_channels": conv_channels,
            "kernel_size": kernel_size,
            "hidden_size": hidden_size,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be a finite number in [0, 1)")

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.dropout_probability = float(dropout)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(1 if index == 0 else conv_channels, conv_channels, kernel_size, stride=1, padding=0)
            for index in range(4)
        )
        recurrent_input_sizes = (input_channels * conv_channels, hidden_size)
        self.input_weights = nn.ParameterList(
            nn.Parameter(torch.empty(4 * hidden_size, size)) for size in recurrent_input_sizes
        )
        self.hidden_weights = nn.ParameterList(
            nn.Parameter(torch.empty(4 * hidden_size, hidden_size)) for _ in range(2)
        )
        self.recurrent_biases = nn.ParameterList(
            nn.Parameter(torch.zeros(4 * hidden_size)) for _ in range(2)
        )
        self.peepholes = nn.ParameterList(
            nn.Parameter(torch.zeros(3, hidden_size)) for _ in range(2)
        )
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        for convolution in self.convolutions:
            nn.init.orthogonal_(convolution.weight)
            nn.init.zeros_(convolution.bias)
        for layer in range(2):
            for gate in range(4):
                start = gate * self.hidden_size
                end = start + self.hidden_size
                nn.init.orthogonal_(self.input_weights[layer][start:end])
                nn.init.orthogonal_(self.hidden_weights[layer][start:end])
            nn.init.zeros_(self.recurrent_biases[layer])
            nn.init.zeros_(self.peepholes[layer])
        nn.init.orthogonal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def _lstm(self, sequence, layer):
        batch_size = sequence.shape[0]
        hidden = sequence.new_zeros(batch_size, self.hidden_size)
        cell = sequence.new_zeros(batch_size, self.hidden_size)
        sequence = self.dropout(sequence)
        outputs = []
        for current in sequence.unbind(dim=1):
            gates = F.linear(current, self.input_weights[layer], self.recurrent_biases[layer])
            gates = gates + F.linear(hidden, self.hidden_weights[layer])
            input_gate, forget_gate, candidate, output_gate = gates.chunk(4, dim=-1)
            input_gate = torch.sigmoid(input_gate + self.peepholes[layer][0] * cell)
            forget_gate = torch.sigmoid(forget_gate + self.peepholes[layer][1] * cell)
            candidate = torch.tanh(candidate)
            cell = forget_gate * cell + input_gate * candidate
            output_gate = torch.sigmoid(output_gate + self.peepholes[layer][2] * cell)
            hidden = output_gate * torch.tanh(cell)
            outputs.append(hidden)
        return torch.stack(outputs, dim=1)

    def forward(self, x):
        if not torch.is_tensor(x) or x.ndim != 3:
            raise ValueError("input must have shape (batch, time, channels)")
        if x.shape[0] < 1 or x.shape[1] < 1:
            raise ValueError("input batch and time dimensions must be nonempty")
        if x.shape[2] != self.input_channels:
            raise ValueError(f"input channels must equal input_channels ({self.input_channels})")
        minimum_time = 4 * (self.kernel_size - 1) + 1
        if x.shape[1] < minimum_time:
            raise ValueError(f"input time dimension must be at least {minimum_time} for four valid convolutions")

        batch_size, _, channels = x.shape
        features = x.permute(0, 2, 1).reshape(batch_size * channels, 1, x.shape[1])
        for convolution in self.convolutions:
            features = torch.relu(convolution(features))
        steps = features.shape[-1]
        features = features.reshape(batch_size, channels, self.conv_channels, steps)
        features = features.permute(0, 3, 1, 2).reshape(batch_size, steps, channels * self.conv_channels)
        recurrent = self._lstm(features, 0)
        recurrent = self._lstm(recurrent, 1)
        logits = self.classifier(self.dropout(recurrent))
        return torch.softmax(logits, dim=-1)
