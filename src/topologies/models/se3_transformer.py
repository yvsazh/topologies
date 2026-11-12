import torch
from torch import nn


class SE3Transformer(nn.Module):
    architecture_id = "se3-transformer"
    PRESETS = {
        "tiny": {
            "in_channels": 1,
            "hidden_channels": 2,
            "out_channels": 1,
            "num_layers": 1,
            "max_degree": 0,
            "radial_hidden": 4,
            "num_heads": 1,
            "self_interaction": "linear",
        },
        "small": {
            "in_channels": 4,
            "hidden_channels": 8,
            "out_channels": 4,
            "num_layers": 2,
            "max_degree": 1,
            "radial_hidden": 16,
            "num_heads": 1,
            "self_interaction": "linear",
        },
        "base": {
            "in_channels": 8,
            "hidden_channels": 16,
            "out_channels": 8,
            "num_layers": 4,
            "max_degree": 1,
            "radial_hidden": 32,
            "num_heads": 1,
            "self_interaction": "linear",
        },
    }

    def __init__(
        self,
        in_channels=1,
        hidden_channels=8,
        out_channels=1,
        num_layers=2,
        max_degree=1,
        radial_hidden=16,
        num_heads=1,
        self_interaction="linear",
    ):
        super().__init__()
        channel_sizes = {
            "in_channels": in_channels,
            "hidden_channels": hidden_channels,
            "out_channels": out_channels,
            "num_layers": num_layers,
            "radial_hidden": radial_hidden,
            "num_heads": num_heads,
        }
        for name, value in channel_sizes.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(max_degree, bool) or not isinstance(max_degree, int) or max_degree not in (0, 1):
            raise ValueError("max_degree must be 0 or 1")
        if hidden_channels % num_heads != 0:
            raise ValueError("hidden_channels must be divisible by num_heads")
        if self_interaction != "linear":
            raise ValueError("self_interaction must be 'linear'")
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.max_degree = max_degree
        self.radial_hidden = radial_hidden
        self.num_heads = num_heads
        self.self_interaction = self_interaction
        self.head_channels = hidden_channels // num_heads
        self.input_projection = nn.Linear(in_channels, hidden_channels, bias=False)
        self.query_projections = nn.ModuleList(
            nn.Linear(hidden_channels, hidden_channels, bias=False) for _ in range(num_layers)
        )
        self.key_projections = nn.ModuleList(
            nn.Linear(hidden_channels, hidden_channels, bias=False) for _ in range(num_layers)
        )
        self.value_projections = nn.ModuleList(
            nn.Linear(hidden_channels, hidden_channels, bias=False) for _ in range(num_layers)
        )
        self.scalar_self_interactions = nn.ModuleList(
            nn.Linear(hidden_channels, hidden_channels, bias=False) for _ in range(num_layers)
        )
        self.radial_functions = nn.ModuleList()
        for _ in range(num_layers):
            radial = nn.Sequential(
                nn.Linear(1, radial_hidden),
                nn.SiLU(),
                nn.Linear(radial_hidden, hidden_channels * 5),
            )
            nn.init.zeros_(radial[-1].weight)
            nn.init.zeros_(radial[-1].bias)
            self.radial_functions.append(radial)
        if max_degree == 1:
            self.vector_self_interactions = nn.ModuleList(
                nn.Linear(hidden_channels, hidden_channels, bias=False)
                for _ in range(num_layers - 1)
            )
            self.scalar_to_vector_projections = nn.ModuleList(
                nn.Linear(hidden_channels, hidden_channels, bias=False)
                for _ in range(num_layers - 1)
            )
            self.vector_to_scalar_projections = nn.ModuleList(
                nn.Linear(hidden_channels, hidden_channels, bias=False)
                for _ in range(num_layers)
            )
            self.vector_to_vector_projections = nn.ModuleList(
                nn.Linear(hidden_channels, hidden_channels, bias=False)
                for _ in range(num_layers - 1)
            )
        else:
            self.vector_self_interactions = nn.ModuleList()
            self.scalar_to_vector_projections = nn.ModuleList()
            self.vector_to_scalar_projections = nn.ModuleList()
            self.vector_to_vector_projections = nn.ModuleList()
        self.output_projection = nn.Linear(hidden_channels, out_channels, bias=False)

    def forward(self, positions, features):
        if not isinstance(positions, torch.Tensor) or positions.ndim != 3:
            raise ValueError("positions must be a tensor shaped [B, N, 3]")
        if positions.shape[-1] != 3 or positions.shape[0] < 1 or positions.shape[1] < 1:
            raise ValueError("positions must have non-empty [B, N, 3] dimensions")
        if not isinstance(features, torch.Tensor) or features.ndim != 3:
            raise ValueError("features must be a tensor shaped [B, N, in_channels]")
        if features.shape[0] != positions.shape[0]:
            raise ValueError("features and positions must have matching batch dimensions")
        if features.shape[1] != positions.shape[1]:
            raise ValueError("features and positions must have matching point dimensions")
        if features.shape[2] != self.in_channels:
            raise ValueError("features channel dimension must match in_channels")
        if not positions.is_floating_point() or not features.is_floating_point():
            raise ValueError("positions and features must have floating point dtypes")
        if positions.device != features.device or positions.dtype != features.dtype:
            raise ValueError("positions and features must share dtype and device")
        scalar = self.input_projection(features)
        batch_size, point_count, _ = scalar.shape
        vector = scalar.new_zeros(batch_size, point_count, self.hidden_channels, 3)
        if point_count == 1:
            for index in range(self.num_layers):
                scalar = self.scalar_self_interactions[index](scalar)
                if self.max_degree == 1 and index < self.num_layers - 1:
                    vector = self.vector_self_interactions[index](
                        vector.permute(0, 1, 3, 2)
                    ).permute(0, 1, 3, 2)
        if point_count > 1:
            relative = positions[:, None, :, :] - positions[:, :, None, :]
            distance = torch.linalg.vector_norm(relative, dim=-1)
            direction = relative / distance.clamp_min(torch.finfo(positions.dtype).eps).unsqueeze(-1)
            diagonal = torch.eye(point_count, dtype=torch.bool, device=positions.device)
            for index in range(self.num_layers):
                radial = self.radial_functions[index](distance.unsqueeze(-1))
                key_scale = 1.0 + radial[..., : self.hidden_channels]
                value_scale = 1.0 + radial[..., self.hidden_channels : self.hidden_channels * 2]
                query = self.query_projections[index](scalar).view(
                    batch_size, point_count, self.num_heads, self.head_channels
                ).permute(0, 2, 1, 3)
                key = self.key_projections[index](scalar).view(
                    batch_size, point_count, self.num_heads, self.head_channels
                ).permute(0, 2, 1, 3)
                key_scale = key_scale.view(
                    batch_size,
                    point_count,
                    point_count,
                    self.num_heads,
                    self.head_channels,
                ).permute(0, 3, 1, 2, 4)
                logits = (query.unsqueeze(3) * key.unsqueeze(2) * key_scale).sum(dim=-1)
                attention = logits.masked_fill(diagonal[None, None], float("-inf")).softmax(dim=-1)
                value = self.value_projections[index](scalar).view(
                    batch_size, point_count, self.num_heads, self.head_channels
                ).permute(0, 2, 1, 3)
                value_scale = value_scale.view(
                    batch_size,
                    point_count,
                    point_count,
                    self.num_heads,
                    self.head_channels,
                ).permute(0, 3, 1, 2, 4)
                scalar_message = (
                    attention.unsqueeze(-1) * value.unsqueeze(2) * value_scale
                ).sum(dim=3)
                scalar_message = scalar_message.permute(0, 2, 1, 3).reshape(
                    batch_size, point_count, self.hidden_channels
                )
                scalar_update = self.scalar_self_interactions[index](scalar) + scalar_message
                if self.max_degree == 1:
                    vector_to_scalar = self.vector_to_scalar_projections[index](
                        vector.permute(0, 1, 3, 2)
                    ).permute(0, 1, 3, 2)
                    scalar_vector_message = (
                        vector_to_scalar[:, None, :, :, :] * direction[:, :, :, None, :]
                    ).sum(dim=-1)
                    vector_to_scalar_scale = 1.0 + radial[
                        ..., self.hidden_channels * 3 : self.hidden_channels * 4
                    ]
                    scalar_vector_message = scalar_vector_message * vector_to_scalar_scale
                    scalar_vector_message = scalar_vector_message.view(
                        batch_size,
                        point_count,
                        point_count,
                        self.num_heads,
                        self.head_channels,
                    ).permute(0, 3, 1, 2, 4)
                    scalar_update = scalar_update + (
                        attention.unsqueeze(-1) * scalar_vector_message
                    ).sum(dim=3).permute(0, 2, 1, 3).reshape(
                        batch_size, point_count, self.hidden_channels
                    )
                    if index < self.num_layers - 1:
                        scalar_to_vector = self.scalar_to_vector_projections[index](scalar)
                        scalar_to_vector_pair = (
                            scalar_to_vector[:, None, :, :, None]
                            * direction[:, :, :, None, :]
                        )
                        scalar_to_vector_scale = 1.0 + radial[
                            ..., self.hidden_channels * 2 : self.hidden_channels * 3
                        ]
                        scalar_to_vector_pair = scalar_to_vector_pair * scalar_to_vector_scale[
                            :, :, :, :, None
                        ]
                        vector_to_vector = self.vector_to_vector_projections[index](
                            vector.permute(0, 1, 3, 2)
                        ).permute(0, 1, 3, 2)
                        vector_to_vector_pair = vector_to_vector[:, None, :, :, :].expand(
                            batch_size, point_count, point_count, self.hidden_channels, 3
                        )
                        vector_to_vector_scale = 1.0 + radial[
                            ..., self.hidden_channels * 4 : self.hidden_channels * 5
                        ]
                        vector_to_vector_pair = vector_to_vector_pair * vector_to_vector_scale[
                            :, :, :, :, None
                        ]
                        vector_message = (scalar_to_vector_pair + vector_to_vector_pair).view(
                            batch_size,
                            point_count,
                            point_count,
                            self.num_heads,
                            self.head_channels,
                            3,
                        ).permute(0, 3, 1, 2, 4, 5)
                        vector_update = self.vector_self_interactions[index](
                            vector.permute(0, 1, 3, 2)
                        ).permute(0, 1, 3, 2)
                        vector_update = vector_update + (
                            attention[:, :, :, :, None, None] * vector_message
                        ).sum(dim=3).permute(0, 2, 1, 3, 4).reshape(
                            batch_size, point_count, self.hidden_channels, 3
                        )
                        vector = vector_update
                scalar = scalar_update
        return self.output_projection(scalar)
