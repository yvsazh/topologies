import torch
from torch import nn


class _BernoulliRBM(nn.Module):
    def __init__(self, visible_units, hidden_units):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(visible_units, hidden_units))
        nn.init.normal_(self.weight, mean=0.0, std=0.01)
        self.visible_bias = nn.Parameter(torch.zeros(visible_units))
        self.hidden_bias = nn.Parameter(torch.zeros(hidden_units))

    def hidden_probabilities(self, visible):
        return torch.sigmoid(visible @ self.weight + self.hidden_bias)

    def visible_probabilities(self, hidden):
        return torch.sigmoid(hidden @ self.weight.T + self.visible_bias)

    def train_cd(self, data, learning_rate, k):
        positive_hidden = self.hidden_probabilities(data)
        hidden_state = torch.bernoulli(positive_hidden)
        with torch.no_grad():
            for _ in range(k):
                visible_probability = self.visible_probabilities(hidden_state)
                visible_state = torch.bernoulli(visible_probability)
                negative_hidden_probability = self.hidden_probabilities(visible_state)
                hidden_state = torch.bernoulli(negative_hidden_probability)
        with torch.no_grad():
            batch_size = data.shape[0]
            self.weight.add_(learning_rate * (data.T @ positive_hidden - visible_state.T @ negative_hidden_probability) / batch_size)
            self.visible_bias.add_(learning_rate * (data - visible_state).mean(dim=0))
            self.hidden_bias.add_(learning_rate * (positive_hidden - negative_hidden_probability).mean(dim=0))


class DBN(nn.Module):
    architecture_id = "dbn"
    def __init__(self, layer_sizes=(4, 3, 2)):
        super().__init__()
        sizes = tuple(layer_sizes)
        if len(sizes) < 2 or any(not isinstance(size, int) or size <= 0 for size in sizes):
            raise ValueError("layer_sizes must contain at least two positive integers")
        self.layer_sizes = sizes
        self.rbms = nn.ModuleList(_BernoulliRBM(left, right) for left, right in zip(sizes, sizes[1:]))

    def encode(self, inputs):
        if inputs.shape[-1] != self.layer_sizes[0]:
            raise ValueError("input feature dimension must match the first layer")
        code = inputs
        for rbm in self.rbms:
            code = rbm.hidden_probabilities(code)
        return code

    def pretrain(self, data, epochs=1, learning_rate=0.1, k=1):
        if data.ndim != 2 or data.shape[1] != self.layer_sizes[0]:
            raise ValueError("data must be a batch with the first layer's feature dimension")
        if epochs <= 0 or learning_rate <= 0 or k <= 0:
            raise ValueError("epochs, learning_rate, and k must be positive")
        with torch.no_grad():
            representation = data
            for rbm in self.rbms:
                for _ in range(epochs):
                    rbm.train_cd(representation, learning_rate, k)
                representation = rbm.hidden_probabilities(representation)
        return self

    def forward(self, inputs):
        return self.encode(inputs)
