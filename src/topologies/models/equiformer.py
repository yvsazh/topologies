import math

import torch
import torch.nn as nn
import torch.nn.functional as F

PRESETS = {
    "tiny": {
        "num_layers": 2,
        "embedding_channels": (8, 4, 2),
        "num_heads": 2,
        "head_channels": (4, 2, 1),
        "ffn_channels": (16, 8, 4),
        "feature_channels": 16,
        "num_species": 5,
        "cutoff": 5.0,
        "num_radial_basis": 8,
        "radial_basis": "gaussian",
        "radial_hidden": (16, 16),
        "nonlinear_message": True,
        "norm": "layer",
        "norm_eps": 1e-5,
        "normalization": "component",
        "attention_dropout": 0.0,
        "projection_dropout": 0.0,
        "output_dropout": 0.0,
        "drop_path": 0.0,
        "avg_degree": 15.57930850982666,
        "avg_num_nodes": 18.03065905448718,
    },
    "small": {
        "num_layers": 4,
        "embedding_channels": (32, 16, 8),
        "num_heads": 4,
        "head_channels": (8, 4, 2),
        "ffn_channels": (64, 32, 16),
        "feature_channels": 64,
        "num_species": 5,
        "cutoff": 5.0,
        "num_radial_basis": 32,
        "radial_basis": "gaussian",
        "radial_hidden": (32, 32),
        "nonlinear_message": True,
        "norm": "layer",
        "norm_eps": 1e-5,
        "normalization": "component",
        "attention_dropout": 0.2,
        "projection_dropout": 0.0,
        "output_dropout": 0.0,
        "drop_path": 0.0,
        "avg_degree": 15.57930850982666,
        "avg_num_nodes": 18.03065905448718,
    },
    "base": {
        "num_layers": 6,
        "embedding_channels": (128, 64, 32),
        "num_heads": 4,
        "head_channels": (32, 16, 8),
        "ffn_channels": (384, 192, 96),
        "feature_channels": 512,
        "num_species": 5,
        "cutoff": 5.0,
        "num_radial_basis": 128,
        "radial_basis": "gaussian",
        "radial_hidden": (64, 64),
        "nonlinear_message": True,
        "norm": "layer",
        "norm_eps": 1e-5,
        "normalization": "component",
        "attention_dropout": 0.2,
        "projection_dropout": 0.0,
        "output_dropout": 0.0,
        "drop_path": 0.0,
        "avg_degree": 15.57930850982666,
        "avg_num_nodes": 18.03065905448718,
    },
}


class Equiformer(nn.Module):
    architecture_id = "equiformer"
    PRESETS = PRESETS

    PATHS = (
        ("scalar_scalar", 0, 0),
        ("scalar_vector", 1, 0),
        ("scalar_tensor", 2, 0),
        ("vector_scalar", 0, 1),
        ("vector_vector_scalar", 1, 1),
        ("vector_vector_cross", 1, 1),
        ("vector_from_vector_rank2", 1, 1),
        ("vector_tensor", 2, 1),
        ("vector_tensor_tensor", 2, 1),
        ("tensor_scalar", 0, 2),
        ("tensor_vector", 1, 2),
        ("tensor_from_tensor_vector", 2, 2),
        ("tensor_tensor_scalar", 2, 2),
        ("tensor_tensor_product", 2, 2),
    )

    @staticmethod
    def _positive_float(name, value):
        if value is True or value is False:
            raise ValueError(f"{name} must be a finite positive number")
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{name} must be a finite positive number") from None
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number")
        return value

    @staticmethod
    def _dropout_value(name, value):
        if value is True or value is False:
            raise ValueError(f"{name} must be in [0, 1)")
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{name} must be in [0, 1)") from None
        if not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f"{name} must be in [0, 1)")
        return value

    def __init__(
        self,
        num_layers=6,
        embedding_channels=(128, 64, 32),
        num_heads=4,
        head_channels=(32, 16, 8),
        ffn_channels=(384, 192, 96),
        feature_channels=512,
        num_species=5,
        cutoff=5.0,
        num_radial_basis=128,
        radial_basis="gaussian",
        radial_hidden=(64, 64),
        nonlinear_message=True,
        norm="layer",
        norm_eps=1e-5,
        normalization="component",
        attention_dropout=0.2,
        projection_dropout=0.0,
        output_dropout=0.0,
        drop_path=0.0,
        avg_degree=15.57930850982666,
        avg_num_nodes=18.03065905448718,
    ):
        super().__init__()
        integer_values = {
            "num_layers": num_layers,
            "num_heads": num_heads,
            "feature_channels": feature_channels,
            "num_species": num_species,
            "num_radial_basis": num_radial_basis,
        }
        for name, value in integer_values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
            ("embedding_channels", embedding_channels),
            ("head_channels", head_channels),
            ("ffn_channels", ffn_channels),
        ):
            if not isinstance(value, (tuple, list)) or len(value) != 3 or any(
                isinstance(channel, bool) or not isinstance(channel, int) or channel < 1
                for channel in value
            ):
                raise ValueError(f"{name} must contain three positive integers")
        if not isinstance(radial_hidden, (tuple, list)) or len(radial_hidden) != 2 or any(
            isinstance(channel, bool) or not isinstance(channel, int) or channel < 1
            for channel in radial_hidden
        ):
            raise ValueError("radial_hidden must contain two positive integers")
        cutoff = self._positive_float("cutoff", cutoff)
        norm_eps = self._positive_float("norm_eps", norm_eps)
        avg_degree = self._positive_float("avg_degree", avg_degree)
        avg_num_nodes = self._positive_float("avg_num_nodes", avg_num_nodes)
        attention_dropout, projection_dropout, output_dropout, drop_path = (
            self._dropout_value(name, value)
            for name, value in (
            ("attention_dropout", attention_dropout),
            ("projection_dropout", projection_dropout),
            ("output_dropout", output_dropout),
            ("drop_path", drop_path),
            )
        )
        if radial_basis != "gaussian":
            raise ValueError("radial_basis must be 'gaussian'")
        if norm != "layer":
            raise ValueError("norm must be 'layer'")
        if normalization != "component":
            raise ValueError("normalization must be 'component'")
        if nonlinear_message is not True and nonlinear_message is not False:
            raise ValueError("nonlinear_message must be a boolean")
        self.num_layers = num_layers
        self.embedding_channels = tuple(embedding_channels)
        self.num_heads = num_heads
        self.head_channels = tuple(head_channels)
        self.ffn_channels = tuple(ffn_channels)
        self.feature_channels = feature_channels
        self.num_species = num_species
        self.cutoff = cutoff
        self.num_radial_basis = num_radial_basis
        self.radial_basis = radial_basis
        self.nonlinear_message = nonlinear_message
        self.norm_eps = norm_eps
        self.attention_dropout = attention_dropout
        self.projection_dropout = projection_dropout
        self.output_dropout = output_dropout
        self.drop_path = drop_path
        self.avg_degree = avg_degree
        self.avg_num_nodes = avg_num_nodes
        if num_species != 5:
            raise ValueError("num_species must be 5 for the QM9 atom vocabulary")
        self.register_buffer("atomic_numbers", torch.tensor([1, 6, 7, 8, 9], dtype=torch.long))
        self.atom_embedding = nn.Embedding(num_species, embedding_channels[0])
        self.initial_source = nn.ModuleList(
            [
                nn.Identity(),
                nn.Linear(embedding_channels[0], embedding_channels[1], bias=False),
                nn.Linear(embedding_channels[0], embedding_channels[2], bias=False),
            ]
        )
        initial_output = sum(embedding_channels)
        self.initial_profile = nn.Sequential(
            nn.Linear(num_radial_basis, radial_hidden[0]),
            nn.LayerNorm(radial_hidden[0]),
            nn.SiLU(),
            nn.Linear(radial_hidden[0], radial_hidden[1]),
            nn.LayerNorm(radial_hidden[1]),
            nn.SiLU(),
            nn.Linear(radial_hidden[1], initial_output),
        )
        self.initial_scalar_norm_weight = nn.Parameter(torch.ones(embedding_channels[0]))
        self.initial_scalar_norm_bias = nn.Parameter(torch.zeros(embedding_channels[0]))
        self.layers = nn.ModuleList()
        self.norm1_scalar_weight = nn.ParameterList()
        self.norm1_scalar_bias = nn.ParameterList()
        self.norm1_vector_scale = nn.ParameterList()
        self.norm1_tensor_scale = nn.ParameterList()
        self.norm2_scalar_weight = nn.ParameterList()
        self.norm2_scalar_bias = nn.ParameterList()
        self.norm2_vector_scale = nn.ParameterList()
        self.norm2_tensor_scale = nn.ParameterList()
        self.attention_vectors = nn.ParameterList()
        profile_size = sum(num_heads * head_channels[out_degree] for _, _, out_degree in self.PATHS)
        for _ in range(num_layers):
            source_projections = nn.ModuleDict()
            destination_projections = nn.ModuleDict()
            message_projections = nn.ModuleDict()
            message_coefficients = nn.ParameterDict()
            for name, in_degree, out_degree in self.PATHS:
                source_projections[name] = nn.Linear(
                    embedding_channels[in_degree],
                    num_heads * head_channels[out_degree],
                    bias=in_degree == 0,
                )
                destination_projections[name] = nn.Linear(
                    embedding_channels[in_degree],
                    num_heads * head_channels[out_degree],
                    bias=in_degree == 0,
                )
                message_projections[name] = nn.Linear(
                    head_channels[in_degree],
                    head_channels[out_degree],
                    bias=in_degree == 0,
                )
                message_coefficients[name] = nn.Parameter(
                    torch.ones(num_heads, head_channels[out_degree])
                )
            layer = nn.ModuleDict(
                {
                    "profile": nn.Sequential(
                        nn.Linear(num_radial_basis, radial_hidden[0]),
                        nn.LayerNorm(radial_hidden[0]),
                        nn.SiLU(),
                        nn.Linear(radial_hidden[0], radial_hidden[1]),
                        nn.LayerNorm(radial_hidden[1]),
                        nn.SiLU(),
                        nn.Linear(radial_hidden[1], profile_size),
                    ),
                    "source_projections": source_projections,
                    "destination_projections": destination_projections,
                    "message_projections": message_projections,
                    "message_coefficients": message_coefficients,
                    "message_gates": nn.ModuleList(
                        [
                            nn.Linear(head_channels[0], head_channels[degree])
                            for degree in range(1, 3)
                        ]
                    ),
                    "message_output_projections": nn.ModuleList(
                        [
                            nn.Linear(head_channels[degree], head_channels[degree], bias=degree == 0)
                            for degree in range(3)
                        ]
                    ),
                    "output_projections": nn.ModuleList(
                        [
                            nn.Linear(num_heads * head_channels[degree], embedding_channels[degree], bias=degree == 0)
                            for degree in range(3)
                        ]
                    ),
                    "ffn_in": nn.ModuleList(
                        [
                            nn.Linear(embedding_channels[degree], ffn_channels[degree], bias=degree == 0)
                            for degree in range(3)
                        ]
                    ),
                    "ffn_gate": nn.ModuleList(
                        [
                            nn.Linear(ffn_channels[0], ffn_channels[degree], bias=True)
                            for degree in range(1, 3)
                        ]
                    ),
                    "ffn_out": nn.ModuleList(
                        [
                            nn.Linear(ffn_channels[degree], embedding_channels[degree], bias=degree == 0)
                            for degree in range(3)
                        ]
                    ),
                }
            )
            self.layers.append(layer)
            self.norm1_scalar_weight.append(nn.Parameter(torch.ones(embedding_channels[0])))
            self.norm1_scalar_bias.append(nn.Parameter(torch.zeros(embedding_channels[0])))
            self.norm1_vector_scale.append(nn.Parameter(torch.ones(embedding_channels[1])))
            self.norm1_tensor_scale.append(nn.Parameter(torch.ones(embedding_channels[2])))
            self.norm2_scalar_weight.append(nn.Parameter(torch.ones(embedding_channels[0])))
            self.norm2_scalar_bias.append(nn.Parameter(torch.zeros(embedding_channels[0])))
            self.norm2_vector_scale.append(nn.Parameter(torch.ones(embedding_channels[1])))
            self.norm2_tensor_scale.append(nn.Parameter(torch.ones(embedding_channels[2])))
            self.attention_vectors.append(nn.Parameter(torch.empty(num_heads, head_channels[0])))
        for attention_vector in self.attention_vectors:
            nn.init.normal_(attention_vector, std=head_channels[0] ** -0.5)
        self.final_norm_weight = nn.Parameter(torch.ones(embedding_channels[0]))
        self.final_norm_bias = nn.Parameter(torch.zeros(embedding_channels[0]))
        self.readout_in = nn.Linear(embedding_channels[0], feature_channels)
        self.readout_out = nn.Linear(feature_channels, 1)
        centers = torch.linspace(0.0, self.cutoff, num_radial_basis)
        self.register_buffer("radial_centers", centers)
        self.radial_width = self.cutoff / max(num_radial_basis - 1, 1)
        self.register_buffer("identity", torch.eye(3))

    @staticmethod
    def _scalar_layer_norm(values, weight, bias, eps):
        mean = values.mean(dim=-1, keepdim=True)
        variance = (values - mean).square().mean(dim=-1, keepdim=True)
        normalized = (values - mean) * torch.rsqrt(variance + eps)
        return normalized * weight + bias

    @staticmethod
    def _raise_input_type_error():
        raise ValueError("species and positions must be tensors")

    @staticmethod
    def _channel_linear(layer, values):
        if values.ndim == 2:
            return layer(values)
        if values.ndim == 3:
            return layer(values.transpose(1, 2)).transpose(1, 2)
        return layer(values.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    @staticmethod
    def _channel_dropout(values, probability, training):
        if probability == 0.0 or not training:
            return values
        shape = values.shape[:2] + (1,) * (values.ndim - 2)
        mask = F.dropout(torch.ones(shape, dtype=values.dtype, device=values.device), probability, True)
        return values * mask

    @staticmethod
    def _drop_path(values, probability, training):
        if probability == 0.0 or not training:
            return values
        shape = (values.shape[0],) + (1,) * (values.ndim - 1)
        keep = 1.0 - probability
        mask = torch.empty(shape, dtype=values.dtype, device=values.device).bernoulli_(keep)
        return values * mask / keep

    @staticmethod
    def _attention_aggregate(logits, values, destination, num_nodes):
        weights = torch.zeros_like(logits)
        for node in range(num_nodes):
            selected = destination == node
            if selected.any():
                weights[selected] = torch.softmax(logits[selected], dim=0)
        aggregate = values.new_zeros((num_nodes,) + values.shape[1:])
        if destination.numel():
            weight_shape = (weights.shape[0], weights.shape[1]) + (1,) * (values.ndim - 2)
            aggregate.index_add_(0, destination, values * weights.view(weight_shape))
        return aggregate, weights

    def _normalize(self, values, scalar_weight, scalar_bias, vector_scale, tensor_scale):
        scalar = self._scalar_layer_norm(
            values[0], scalar_weight, scalar_bias, self.norm_eps
        )
        vector_rms = torch.sqrt(values[1].square().sum(dim=-1).mean(dim=-1, keepdim=True) + self.norm_eps)
        vector = values[1] / vector_rms.unsqueeze(-1) * vector_scale.view(1, -1, 1)
        tensor_rms = torch.sqrt(
            values[2].square().sum(dim=(-1, -2)).mean(dim=-1, keepdim=True) + self.norm_eps
        )
        tensor = values[2] / tensor_rms.unsqueeze(-1).unsqueeze(-1) * tensor_scale.view(1, -1, 1, 1)
        return [scalar, vector, tensor]

    def _radial_features(self, distance):
        delta = (distance.unsqueeze(-1) - self.radial_centers) / max(self.radial_width, 1e-6)
        basis = torch.exp(-0.5 * delta.square())
        envelope = 0.5 * (torch.cos(math.pi * distance / self.cutoff) + 1.0)
        return basis * envelope.unsqueeze(-1)

    @staticmethod
    def _head_channel_linear(layer, values):
        if values.ndim == 3:
            return layer(values)
        axes = (0, 1) + tuple(range(3, values.ndim)) + (2,)
        inverse = tuple(axes.index(index) for index in range(values.ndim))
        return layer(values.permute(axes)).permute(inverse)

    def _path_product(self, name, feature, sh1, sh2):
        if name == "scalar_scalar" or name == "vector_vector_scalar" or name == "tensor_tensor_scalar":
            return feature
        if name == "scalar_vector":
            return torch.einsum("ehdi,ei->ehd", feature, sh1)
        if name == "scalar_tensor":
            return torch.einsum("ehdij,eij->ehd", feature, sh2)
        if name == "vector_scalar":
            return feature.unsqueeze(-1) * sh1[:, None, None, :]
        if name == "vector_vector_cross":
            return torch.cross(feature, sh1[:, None, None, :].expand_as(feature), dim=-1)
        if name == "vector_from_vector_rank2":
            return torch.einsum("eij,ehdj->ehdi", sh2, feature)
        if name == "vector_tensor":
            return torch.einsum("ehdij,ej->ehdi", feature, sh1)
        if name == "vector_tensor_tensor":
            edge_tensor = sh2[:, None, None, :, :]
            commutator = torch.matmul(feature, edge_tensor) - torch.matmul(edge_tensor, feature)
            return 0.5 * torch.stack(
                (commutator[..., 2, 1], commutator[..., 0, 2], commutator[..., 1, 0]), dim=-1
            )
        if name == "tensor_scalar":
            return feature.unsqueeze(-1).unsqueeze(-1) * sh2[:, None, None, :, :]
        if name == "tensor_vector":
            left = feature.unsqueeze(-1) * sh1[:, None, None, None, :]
            right = sh1[:, None, None, :, None] * feature.unsqueeze(-2)
            contribution = 0.5 * (left + right)
            trace = contribution.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / 3.0
            return contribution - trace[..., None, None] * self.identity
        if name == "tensor_from_tensor_vector":
            zero = torch.zeros_like(sh1[:, 0])
            skew = torch.stack(
                (
                    torch.stack((zero, -sh1[:, 2], sh1[:, 1]), dim=-1),
                    torch.stack((sh1[:, 2], zero, -sh1[:, 0]), dim=-1),
                    torch.stack((-sh1[:, 1], sh1[:, 0], zero), dim=-1),
                ),
                dim=-2,
            )[:, None, None, :, :]
            commutator = torch.matmul(skew, feature) - torch.matmul(feature, skew)
            contribution = 0.5 * (commutator + commutator.transpose(-1, -2))
            trace = contribution.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / 3.0
            return contribution - trace[..., None, None] * self.identity
        edge_tensor = sh2[:, None, None, :, :]
        product = 0.5 * (torch.matmul(feature, edge_tensor) + torch.matmul(edge_tensor, feature))
        trace = product.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / 3.0
        return product - trace[..., None, None] * self.identity

    def _geometric_paths(self, layer, values, source, destination, radial, sh1, sh2):
        edge_count = source.numel()
        output_values = [None, None, None]
        offset = 0
        for name, in_degree, out_degree in self.PATHS:
            width = self.num_heads * self.head_channels[out_degree]
            source_feature = self._channel_linear(
                layer["source_projections"][name], values[in_degree][source]
            )
            destination_feature = self._channel_linear(
                layer["destination_projections"][name], values[in_degree][destination]
            )
            feature = source_feature + destination_feature
            feature = feature.reshape(
                (edge_count, self.num_heads, self.head_channels[out_degree])
                + tuple(values[in_degree].shape[2:])
            )
            coefficient = radial[:, offset : offset + width].reshape(
                edge_count, self.num_heads, self.head_channels[out_degree]
            )
            offset += width
            contribution = self._path_product(name, feature, sh1, sh2)
            factor_shape = coefficient.shape + (1,) * (contribution.ndim - 3)
            contribution = contribution * coefficient.view(factor_shape)
            if output_values[out_degree] is None:
                output_values[out_degree] = contribution
            else:
                output_values[out_degree] = output_values[out_degree] + contribution
        return output_values

    def _nonlinear_values(self, layer, features, sh1, sh2):
        values = [F.silu(features[0])]
        for degree in range(1, 3):
            gate = torch.sigmoid(layer["message_gates"][degree - 1](features[0]))
            values.append(features[degree] * gate.unsqueeze(-1) if degree == 1 else
                          features[degree] * gate.unsqueeze(-1).unsqueeze(-1))
        output_values = [None, None, None]
        for name, in_degree, out_degree in self.PATHS:
            projected = self._head_channel_linear(
                layer["message_projections"][name], values[in_degree]
            )
            contribution = self._path_product(name, projected, sh1, sh2)
            coefficient = layer["message_coefficients"][name]
            factor_shape = (1,) + coefficient.shape + (1,) * (contribution.ndim - 3)
            contribution = contribution * coefficient.view(factor_shape)
            if output_values[out_degree] is None:
                output_values[out_degree] = contribution
            else:
                output_values[out_degree] = output_values[out_degree] + contribution
        return [
            self._head_channel_linear(layer["message_output_projections"][degree], value)
            for degree, value in enumerate(output_values)
        ]

    def _validate_inputs(self, species, positions, batch):
        if not isinstance(species, torch.Tensor) or species.ndim != 1:
            raise ValueError("species must be a one-dimensional integer tensor")
        if species.dtype not in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
            raise ValueError("species must use an integer dtype")
        if not isinstance(positions, torch.Tensor) or positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape [N, 3]")
        if positions.shape[0] != species.shape[0]:
            raise ValueError("species and positions must contain the same number of atoms")
        if species.shape[0] == 0:
            raise ValueError("at least one atom is required")
        if not positions.is_floating_point():
            raise ValueError("positions must use a floating-point dtype")
        if not bool(torch.isfinite(positions).all()):
            raise ValueError("positions must contain only finite values")
        if species.device != positions.device:
            raise ValueError("species and positions must be on the same device")
        supported = (species.unsqueeze(-1) == self.atomic_numbers).any(dim=-1)
        if not bool(supported.all()):
            raise ValueError("species must contain only QM9 atomic numbers 1, 6, 7, 8, and 9")
        if batch is None:
            batch = torch.zeros(species.shape[0], dtype=torch.long, device=species.device)
        if not isinstance(batch, torch.Tensor) or batch.ndim != 1 or batch.shape[0] != species.shape[0]:
            raise ValueError("batch must have shape [N] and match the atom count")
        if batch.dtype not in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
            raise ValueError("batch must use an integer dtype")
        if batch.device != species.device:
            raise ValueError("batch and species must be on the same device")
        if bool((batch < 0).any()):
            raise ValueError("batch graph IDs must be nonnegative")
        graph_count = int(batch.max().item()) + 1
        if not torch.equal(
            torch.unique(batch), torch.arange(graph_count, device=batch.device, dtype=batch.dtype)
        ):
            raise ValueError("batch graph IDs must be contiguous from zero")
        category = (species.unsqueeze(-1) == self.atomic_numbers).to(torch.long).argmax(dim=-1)
        return category, positions.to(dtype=self.atom_embedding.weight.dtype), batch.long(), graph_count

    def forward(self, species, positions=None, batch=None):
        if isinstance(species, dict):
            if positions is not None:
                raise ValueError("positions must be omitted when species is an input dictionary")
            inputs = species
            if "species" not in inputs or "positions" not in inputs:
                raise ValueError("input dictionary must contain species and positions")
            species = inputs["species"]
            positions = inputs["positions"]
            batch = inputs.get("batch", batch)
        if positions is None:
            raise ValueError("positions are required")
        if not isinstance(species, torch.Tensor) or not isinstance(positions, torch.Tensor):
            self._raise_input_type_error()
        if species.ndim == 2:
            if positions.ndim != 3 or positions.shape[:2] != species.shape or positions.shape[2] != 3:
                raise ValueError("batched positions must have shape [B, N, 3]")
            graph_count, nodes_per_graph = species.shape
            species = species.reshape(-1)
            positions = positions.reshape(-1, 3)
            if batch is None:
                batch = torch.arange(graph_count, device=species.device).repeat_interleave(nodes_per_graph)
            elif batch.ndim == 2 and batch.shape == (graph_count, nodes_per_graph):
                batch = batch.reshape(-1)
        elif positions.ndim != 2:
            raise ValueError("positions must have shape [N, 3] or [B, N, 3]")
        category, positions, batch, graph_count = self._validate_inputs(species, positions, batch)
        delta = positions[:, None, :] - positions[None, :, :]
        distances = torch.linalg.vector_norm(delta, dim=-1)
        same_graph = batch[:, None] == batch[None, :]
        edge_mask = (distances < self.cutoff) & (distances > 0) & same_graph
        destination, source = torch.where(edge_mask)
        edge_vector = positions[source] - positions[destination]
        edge_distance = distances[destination, source]
        direction = edge_vector / edge_distance.unsqueeze(-1).clamp_min(1e-12)
        spherical_1 = direction
        spherical_2 = direction.unsqueeze(-1) * direction.unsqueeze(-2) - self.identity / 3.0
        radial_basis = self._radial_features(edge_distance)
        node_count = species.shape[0]
        scalar = self.atom_embedding(category)
        vector = positions.new_zeros((node_count, self.embedding_channels[1], 3))
        tensor = positions.new_zeros((node_count, self.embedding_channels[2], 3, 3))
        initial_radial = self.initial_profile(radial_basis)
        offset = 0
        node_values = [scalar, vector, tensor]
        for degree, channel_count in enumerate(self.embedding_channels):
            coefficients = initial_radial[:, offset : offset + channel_count]
            offset += channel_count
            source_scalar = self.initial_source[degree](scalar[source])
            if degree == 0:
                messages = source_scalar * coefficients
            elif degree == 1:
                messages = source_scalar.unsqueeze(-1) * coefficients.unsqueeze(-1) * spherical_1[:, None, :]
            else:
                messages = (
                    source_scalar.unsqueeze(-1).unsqueeze(-1)
                    * coefficients.unsqueeze(-1).unsqueeze(-1)
                    * spherical_2[:, None, :, :]
                )
            node_values[degree] = node_values[degree].index_add(0, destination, messages)
        node_values[0] = node_values[0] / math.sqrt(self.avg_degree)
        node_values[1] = node_values[1] / math.sqrt(self.avg_degree)
        node_values[2] = node_values[2] / math.sqrt(self.avg_degree)
        node_values[0] = node_values[0] + scalar
        for layer_index, layer in enumerate(self.layers):
            normalized = self._normalize(
                node_values,
                self.norm1_scalar_weight[layer_index],
                self.norm1_scalar_bias[layer_index],
                self.norm1_vector_scale[layer_index],
                self.norm1_tensor_scale[layer_index],
            )
            radial = layer["profile"](radial_basis)
            message_values = self._geometric_paths(
                layer, normalized, source, destination, radial, spherical_1, spherical_2
            )
            attention_logits = message_values[0]
            attention_logits = 0.5 * (1.2 * attention_logits + 0.8 * torch.sqrt(attention_logits.square() + 1e-6))
            attention_logits = (attention_logits * self.attention_vectors[layer_index]).sum(dim=-1)
            value_messages = (
                self._nonlinear_values(layer, message_values, spherical_1, spherical_2)
                if self.nonlinear_message
                else message_values
            )
            aggregates = []
            weights = None
            for degree, values in enumerate(value_messages):
                aggregate, weights = self._attention_aggregate(
                    attention_logits,
                    values,
                    destination,
                    node_count,
                )
                if self.training and self.attention_dropout:
                    dropped_weights = F.dropout(weights, self.attention_dropout, True)
                    weight_shape = (dropped_weights.shape[0], dropped_weights.shape[1]) + (
                        1,
                    ) * (values.ndim - 2)
                    aggregate = values.new_zeros((node_count,) + values.shape[1:])
                    if destination.numel():
                        aggregate.index_add_(
                            0,
                            destination,
                            values * dropped_weights.view(weight_shape),
                        )
                flattened = aggregate.reshape(
                    (node_count, self.num_heads * self.head_channels[degree])
                    + tuple(aggregate.shape[3:])
                )
                projected = self._channel_linear(layer["output_projections"][degree], flattened)
                projected = self._channel_dropout(projected, self.projection_dropout, self.training)
                aggregates.append(projected)
            node_values = [
                current + self._drop_path(update, self.drop_path, self.training)
                for current, update in zip(node_values, aggregates)
            ]
            normalized = self._normalize(
                node_values,
                self.norm2_scalar_weight[layer_index],
                self.norm2_scalar_bias[layer_index],
                self.norm2_vector_scale[layer_index],
                self.norm2_tensor_scale[layer_index],
            )
            scalar_hidden = F.relu(layer["ffn_in"][0](normalized[0]))
            updates = [layer["ffn_out"][0](scalar_hidden)]
            for degree in range(1, 3):
                hidden = self._channel_linear(layer["ffn_in"][degree], normalized[degree])
                gate = torch.sigmoid(layer["ffn_gate"][degree - 1](scalar_hidden))
                hidden = hidden * gate.view((node_count, self.ffn_channels[degree]) + (1,) * (hidden.ndim - 2))
                updates.append(self._channel_linear(layer["ffn_out"][degree], hidden))
            node_values = [
                current + self._drop_path(
                    self._channel_dropout(update, self.output_dropout, self.training),
                    self.drop_path,
                    self.training,
                )
                for current, update in zip(node_values, updates)
            ]
        scalar_features = self._scalar_layer_norm(
            node_values[0], self.final_norm_weight, self.final_norm_bias, self.norm_eps
        )
        scalar_features = F.silu(self.readout_in(scalar_features))
        pooled = scalar_features.new_zeros((graph_count, self.feature_channels))
        pooled.index_add_(0, batch, scalar_features)
        pooled = pooled / math.sqrt(self.avg_num_nodes)
        pooled = F.dropout(pooled, self.output_dropout, self.training)
        return self.readout_out(pooled)
