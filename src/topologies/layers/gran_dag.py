import math
import numbers

import torch
from torch import nn


class GranDAG(nn.Module):
    component_id = "gran-dag"
    PRESETS = {
        "tiny": {"num_variables": 3, "hidden_dim": 4, "hidden_layers": 1},
        "small": {"num_variables": 5, "hidden_dim": 8, "hidden_layers": 2},
        "base": {"num_variables": 10, "hidden_dim": 10, "hidden_layers": 2},
    }

    def __init__(self, num_variables, hidden_dim=10, hidden_layers=2, negative_slope=0.01):
        if isinstance(num_variables, bool) or not isinstance(num_variables, numbers.Integral) or num_variables <= 0:
            raise ValueError("num_variables must be a positive integer.")
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, numbers.Integral) or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer.")
        if isinstance(hidden_layers, bool) or not isinstance(hidden_layers, numbers.Integral) or hidden_layers <= 0:
            raise ValueError("hidden_layers must be a positive integer.")
        if not GranDAG._is_finite_real(negative_slope) or negative_slope < 0:
            raise ValueError("negative_slope must be a finite nonnegative real number.")

        super().__init__()
        self.num_variables = int(num_variables)
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = int(hidden_layers)
        self.negative_slope = float(negative_slope)
        self.register_buffer("input_masks", 1.0 - torch.eye(self.num_variables))
        self.networks = nn.ModuleList()

        for _ in range(self.num_variables):
            layers = []
            input_dim = self.num_variables
            for _ in range(self.hidden_layers):
                layers.append(nn.Linear(input_dim, self.hidden_dim))
                layers.append(nn.LeakyReLU(negative_slope=self.negative_slope))
                input_dim = self.hidden_dim
            layers.append(nn.Linear(input_dim, 1, bias=False))
            network = nn.Sequential(*layers)
            for layer in network:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
            self.networks.append(network)

        self.conditional_biases = nn.Parameter(torch.zeros(self.num_variables, 2))

    @staticmethod
    def _is_finite_real(value):
        return not isinstance(value, bool) and isinstance(value, numbers.Real) and math.isfinite(value)

    @property
    def log_variance(self):
        return self.conditional_biases[:, 1]

    def _validate_observations(self, x):
        if not torch.is_tensor(x):
            raise ValueError("observations must be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("observations must have shape [batch, num_variables].")
        if x.shape[0] == 0:
            raise ValueError("observations must contain at least one sample.")
        if x.shape[1] != self.num_variables:
            raise ValueError("observations last dimension must equal num_variables.")
        if not torch.is_floating_point(x):
            raise ValueError("observations must contain real floating-point values.")

    def forward(self, x):
        self._validate_observations(x)
        return torch.cat(
            [
                network(x * self.input_masks[index]) + self.conditional_biases[index, 0]
                for index, network in enumerate(self.networks)
            ],
            dim=1,
        )

    def path_connectivity(self):
        rows = []
        for index, network in enumerate(self.networks):
            linear_layers = [layer for layer in network if isinstance(layer, nn.Linear)]
            connectivity = linear_layers[0].weight.abs()
            for layer in linear_layers[1:]:
                connectivity = layer.weight.abs() @ connectivity
            connectivity = connectivity * self.input_masks[index].abs().unsqueeze(0)
            rows.append(connectivity.sum(dim=0))
        return torch.stack(rows, dim=0)

    def adjacency(self):
        connectivity = self.path_connectivity().transpose(0, 1)
        mask = 1.0 - torch.eye(self.num_variables, dtype=connectivity.dtype, device=connectivity.device)
        return connectivity * mask

    def acyclicity(self, adjacency=None):
        if adjacency is None:
            adjacency = self.adjacency()
        if not torch.is_tensor(adjacency):
            raise ValueError("adjacency must be a torch.Tensor.")
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency must be a square matrix.")
        if adjacency.shape != torch.Size([self.num_variables, self.num_variables]):
            raise ValueError("adjacency dimensions must equal num_variables.")
        if not torch.is_floating_point(adjacency):
            raise ValueError("adjacency must contain real floating-point values.")
        return torch.matrix_exp(adjacency).diagonal().sum() - self.num_variables

    def gaussian_log_likelihood(self, x, means=None):
        self._validate_observations(x)
        if means is None:
            means = self(x)
        if not torch.is_tensor(means) or means.shape != x.shape:
            raise ValueError("means must be a tensor with the same shape as observations.")
        if not torch.is_floating_point(means):
            raise ValueError("means must contain real floating-point values.")
        if means.dtype != x.dtype or means.device != x.device:
            raise ValueError("means and observations must have the same dtype and device.")
        log_two_pi = math.log(2.0 * math.pi)
        log_prob = -0.5 * (
            log_two_pi
            + self.log_variance
            + (x - means).square() * torch.exp(-self.log_variance)
        )
        return log_prob.sum(dim=1).mean()

    def augmented_objective(self, log_likelihood, lagrange_multiplier=0.0, penalty=1.0, acyclicity=None):
        if not torch.is_tensor(log_likelihood) or log_likelihood.ndim != 0:
            raise ValueError("log_likelihood must be a scalar torch.Tensor.")
        if not torch.is_floating_point(log_likelihood):
            raise ValueError("log_likelihood must be a floating-point tensor.")
        if not self._is_finite_real(lagrange_multiplier):
            raise ValueError("lagrange_multiplier must be a finite real number.")
        if not self._is_finite_real(penalty):
            raise ValueError("penalty must be a finite nonnegative real number.")
        if penalty < 0:
            raise ValueError("penalty must be a finite nonnegative real number.")
        if acyclicity is None:
            acyclicity = self.acyclicity()
        elif not torch.is_tensor(acyclicity) or acyclicity.ndim != 0:
            raise ValueError("acyclicity must be a scalar torch.Tensor.")
        elif not torch.is_floating_point(acyclicity):
            raise ValueError("acyclicity must be a floating-point tensor.")
        if acyclicity.device != log_likelihood.device or acyclicity.dtype != log_likelihood.dtype:
            raise ValueError("acyclicity and log_likelihood must have the same dtype and device.")
        return log_likelihood - lagrange_multiplier * acyclicity - 0.5 * penalty * acyclicity.square()
