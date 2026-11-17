import math

import torch
from torch import nn


class STGCN(nn.Module):
    component_id = "stgcn"
    PRESETS = {
        "tiny": {
            "adjacency": [[0.0, 1.0], [1.0, 0.0]],
            "in_channels": 1,
            "temporal_channels": (2, 1, 2),
            "spatial_kernel_size": 1,
            "temporal_kernel_size": 1,
            "dropout": 0.0,
        },
        "small": {
            "adjacency": [
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ],
            "in_channels": 1,
            "temporal_channels": (8, 4, 8),
            "spatial_kernel_size": 3,
            "temporal_kernel_size": 2,
            "dropout": 0.0,
        },
        "base": {
            "adjacency": [
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ],
            "in_channels": 1,
            "temporal_channels": (64, 16, 64),
            "spatial_kernel_size": 3,
            "temporal_kernel_size": 3,
            "dropout": 0.0,
        },
    }

    def __init__(
        self,
        adjacency,
        in_channels=1,
        temporal_channels=(64, 16, 64),
        spatial_kernel_size=3,
        temporal_kernel_size=3,
        dropout=0.0,
        layer_norm_eps=1e-6,
    ):
        super().__init__()
        try:
            graph = torch.as_tensor(adjacency)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError("adjacency must be a square numeric graph") from error
        if graph.ndim != 2 or graph.shape[0] != graph.shape[1] or graph.shape[0] < 1:
            raise ValueError("adjacency must be a nonempty square matrix")
        if graph.is_complex():
            raise ValueError("adjacency must contain real weights")
        graph = graph.detach().to(device="cpu", dtype=torch.get_default_dtype())
        if not torch.isfinite(graph).all():
            raise ValueError("adjacency weights must be finite")
        if (graph < 0).any():
            raise ValueError("adjacency weights must be nonnegative")
        if not torch.allclose(graph, graph.transpose(0, 1), atol=1e-6, rtol=1e-5):
            raise ValueError("adjacency must be symmetric")
        if not self._is_positive_integer(in_channels):
            raise ValueError("in_channels must be a positive integer")
        if (
            not isinstance(temporal_channels, (tuple, list))
            or len(temporal_channels) != 3
            or any(not self._is_positive_integer(value) for value in temporal_channels)
        ):
            raise ValueError("temporal_channels must contain three positive integers")
        if not self._is_positive_integer(spatial_kernel_size):
            raise ValueError("spatial_kernel_size must be a positive integer")
        if not self._is_positive_integer(temporal_kernel_size):
            raise ValueError("temporal_kernel_size must be a positive integer")
        if not isinstance(dropout, (int, float)) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be a finite number in [0, 1)")
        if (
            not isinstance(layer_norm_eps, (int, float))
            or not math.isfinite(layer_norm_eps)
            or layer_norm_eps <= 0
        ):
            raise ValueError("layer_norm_eps must be a positive finite number")

        channels_0, channels_1, channels_2 = temporal_channels
        self.num_nodes = graph.shape[0]
        self.in_channels = in_channels
        self.temporal_channels = tuple(temporal_channels)
        self.spatial_kernel_size = spatial_kernel_size
        self.temporal_kernel_size = temporal_kernel_size
        self.register_buffer("adjacency", graph)
        self.register_buffer(
            "chebyshev_basis", self._make_chebyshev_basis(graph, spatial_kernel_size)
        )
        self.temporal_conv_0 = nn.Conv1d(
            in_channels, 2 * channels_0, temporal_kernel_size
        )
        self.temporal_residual_0 = nn.Linear(in_channels, channels_0)
        self.graph_weight = nn.Parameter(torch.empty(spatial_kernel_size, channels_0, channels_1))
        self.graph_bias = nn.Parameter(torch.zeros(channels_1))
        self.graph_residual = nn.Linear(channels_0, channels_1)
        self.temporal_conv_1 = nn.Conv1d(
            channels_1, 2 * channels_2, temporal_kernel_size
        )
        self.temporal_residual_1 = nn.Linear(channels_1, channels_2)
        self.norm = nn.LayerNorm((self.num_nodes, channels_2), eps=layer_norm_eps)
        self.dropout = nn.Dropout(float(dropout))
        self._initialize_parameters()

    @staticmethod
    def _is_positive_integer(value):
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    @staticmethod
    def _make_chebyshev_basis(adjacency, kernel_size):
        degree = adjacency.sum(dim=1)
        inverse_sqrt_degree = torch.where(
            degree > 0, degree.clamp_min(torch.finfo(adjacency.dtype).tiny).rsqrt(), 0.0
        )
        normalized_adjacency = (
            inverse_sqrt_degree[:, None] * adjacency * inverse_sqrt_degree[None, :]
        )
        identity = torch.eye(adjacency.shape[0], dtype=adjacency.dtype)
        laplacian = identity - normalized_adjacency
        largest_eigenvalue = torch.linalg.eigvalsh(laplacian).amax()
        largest_eigenvalue = torch.where(
            largest_eigenvalue > 0, largest_eigenvalue, torch.ones_like(largest_eigenvalue)
        )
        scaled_laplacian = 2.0 * laplacian / largest_eigenvalue - identity
        basis = [identity]
        if kernel_size > 1:
            basis.append(scaled_laplacian)
        for _ in range(2, kernel_size):
            basis.append(2.0 * scaled_laplacian @ basis[-1] - basis[-2])
        return torch.stack(basis)

    def _initialize_parameters(self):
        modules = (
            self.temporal_conv_0,
            self.temporal_residual_0,
            self.graph_residual,
            self.temporal_conv_1,
            self.temporal_residual_1,
        )
        for module in modules:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.graph_weight)

    def _temporal_glu(self, values, convolution, residual_projection, output_channels):
        batch, time, nodes, channels = values.shape
        sequence = values.permute(0, 2, 3, 1).reshape(batch * nodes, channels, time)
        projected = convolution(sequence)
        projected = projected.reshape(batch, nodes, 2 * output_channels, -1)
        projected = projected.permute(0, 3, 1, 2)
        linear, gate = projected.chunk(2, dim=-1)
        residual = values[:, self.temporal_kernel_size - 1 :, :, :]
        residual = residual_projection(residual)
        return (linear + residual) * torch.sigmoid(gate)

    def forward(self, values):
        if not isinstance(values, torch.Tensor) or values.ndim != 4:
            raise ValueError("input must be a tensor with shape [batch, time, nodes, channels]")
        if values.shape[2] != self.num_nodes:
            raise ValueError("input node count must match adjacency")
        if values.shape[3] != self.in_channels:
            raise ValueError("input channel count must match in_channels")
        min_time = 2 * self.temporal_kernel_size - 1
        if values.shape[1] < min_time:
            raise ValueError(f"input time length must be at least {min_time}")
        if not values.is_floating_point():
            raise ValueError("input values must have a floating-point dtype")

        temporal = self._temporal_glu(
            values, self.temporal_conv_0, self.temporal_residual_0, self.temporal_channels[0]
        )
        spatial = torch.einsum(
            "knm,btmi,kio->btno", self.chebyshev_basis, temporal, self.graph_weight
        )
        spatial = torch.relu(spatial + self.graph_residual(temporal) + self.graph_bias)
        temporal = self._temporal_glu(
            spatial, self.temporal_conv_1, self.temporal_residual_1, self.temporal_channels[2]
        )
        return self.dropout(self.norm(temporal))
