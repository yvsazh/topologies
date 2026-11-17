import math

import torch
from torch import nn
from torch.nn import functional as F


class TokenGT(nn.Module):
    architecture_id = "tokengt"
    PRESETS = {
        "tiny": {
            "input_dim": 4,
            "node_id_dim": 4,
            "hidden_dim": 16,
            "type_id_dim": 16,
            "num_layers": 1,
            "num_heads": 2,
            "mlp_dim": 32,
            "output_dim": 1,
            "dropout": 0.1,
        },
        "small": {
            "input_dim": 16,
            "node_id_dim": 16,
            "hidden_dim": 64,
            "type_id_dim": 64,
            "num_layers": 3,
            "num_heads": 4,
            "mlp_dim": 128,
            "output_dim": 1,
            "dropout": 0.1,
        },
    }

    def __init__(
        self,
        input_dim,
        node_id_dim,
        hidden_dim=64,
        type_id_dim=None,
        num_layers=2,
        num_heads=4,
        mlp_dim=128,
        output_dim=1,
        dropout=0.1,
    ):
        super().__init__()
        integer_arguments = {
            "input_dim": input_dim,
            "node_id_dim": node_id_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "mlp_dim": mlp_dim,
            "output_dim": output_dim,
        }
        for name, value in integer_arguments.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type_id_dim is None:
            type_id_dim = hidden_dim
        if (
            isinstance(type_id_dim, bool)
            or not isinstance(type_id_dim, int)
            or type_id_dim <= 0
        ):
            raise ValueError("type_id_dim must be a positive integer")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (int, float))
            or not math.isfinite(dropout)
            or dropout < 0.0
            or dropout >= 1.0
        ):
            raise ValueError("dropout must be a finite number in [0, 1)")

        self.input_dim = input_dim
        self.node_id_dim = node_id_dim
        self.hidden_dim = hidden_dim
        self.type_id_dim = type_id_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.input_projection = nn.Linear(
            input_dim + 2 * node_id_dim + type_id_dim, hidden_dim
        )
        self.type_identifiers = nn.Parameter(torch.empty(2, type_id_dim))
        self.graph_token = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm1": nn.LayerNorm(hidden_dim),
                        "qkv": nn.Linear(hidden_dim, 3 * hidden_dim),
                        "attention_output": nn.Linear(hidden_dim, hidden_dim),
                        "attention_dropout": nn.Dropout(dropout),
                        "norm2": nn.LayerNorm(hidden_dim),
                        "mlp_in": nn.Linear(hidden_dim, mlp_dim),
                        "mlp_out": nn.Linear(mlp_dim, hidden_dim),
                        "mlp_dropout": nn.Dropout(dropout),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        nn.init.normal_(self.type_identifiers, mean=0.0, std=0.02)
        nn.init.normal_(self.graph_token, mean=0.0, std=0.02)

    def _validate_inputs(
        self, node_features, edge_features, edge_index, node_identifiers
    ):
        if not isinstance(node_features, torch.Tensor) or node_features.ndim != 3:
            raise ValueError("node_features must have shape [batch, nodes, input_dim]")
        if node_features.shape[0] <= 0 or node_features.shape[1] <= 0:
            raise ValueError("node_features must contain a nonempty batch and nodes")
        if node_features.shape[2] != self.input_dim:
            raise ValueError(f"node_features must have feature width {self.input_dim}")
        if not isinstance(edge_features, torch.Tensor) or edge_features.ndim != 3:
            raise ValueError("edge_features must have shape [batch, edges, input_dim]")
        if edge_features.shape[0] != node_features.shape[0]:
            raise ValueError("edge_features and node_features must have the same batch")
        if edge_features.shape[2] != self.input_dim:
            raise ValueError(f"edge_features must have feature width {self.input_dim}")
        if not isinstance(edge_index, torch.Tensor) or edge_index.ndim != 3:
            raise ValueError("edge_index must have shape [batch, 2, edges]")
        if edge_index.shape[0] != node_features.shape[0] or edge_index.shape[1] != 2:
            raise ValueError("edge_index must have shape [batch, 2, edges]")
        if edge_index.shape[2] != edge_features.shape[1]:
            raise ValueError("edge_index and edge_features must contain the same edges")
        if edge_index.dtype != torch.long:
            raise ValueError("edge_index must have dtype torch.long")
        if not isinstance(node_identifiers, torch.Tensor) or node_identifiers.ndim != 3:
            raise ValueError(
                "node_identifiers must have shape [batch, nodes, node_id_dim]"
            )
        if node_identifiers.shape[:2] != node_features.shape[:2]:
            raise ValueError("node_identifiers must match the batch and node dimensions")
        if node_identifiers.shape[2] != self.node_id_dim:
            raise ValueError(
                f"node_identifiers must have feature width {self.node_id_dim}"
            )
        floating_inputs = (node_features, edge_features, node_identifiers)
        if any(not tensor.is_floating_point() for tensor in floating_inputs):
            raise ValueError("node_features, edge_features, and node_identifiers must be floating point")
        reference = self.input_projection.weight
        if any(tensor.device != reference.device for tensor in floating_inputs):
            raise ValueError("floating inputs and model parameters must be on the same device")
        if any(tensor.dtype != reference.dtype for tensor in floating_inputs):
            raise ValueError("floating inputs and model parameters must have the same dtype")
        if edge_index.device != reference.device:
            raise ValueError("edge_index and model parameters must be on the same device")
        if edge_index.numel() > 0:
            sources = edge_index[:, 0]
            destinations = edge_index[:, 1]
            if torch.any(sources < 0) or torch.any(sources >= node_features.shape[1]):
                raise ValueError("edge_index contains an endpoint outside the node range")
            if torch.any(destinations < 0) or torch.any(
                destinations >= node_features.shape[1]
            ):
                raise ValueError("edge_index contains an endpoint outside the node range")

    def _augment_tokens(
        self, node_features, edge_features, edge_index, node_identifiers
    ):
        self._validate_inputs(node_features, edge_features, edge_index, node_identifiers)
        batch_size, num_nodes, _ = node_features.shape
        node_type = self.type_identifiers[0].view(1, 1, self.type_id_dim).expand(
            batch_size, num_nodes, self.type_id_dim
        )
        node_tokens = torch.cat(
            (node_features, node_identifiers, node_identifiers, node_type), dim=-1
        )
        edge_sources = node_identifiers.gather(
            1,
            edge_index[:, 0].unsqueeze(-1).expand(-1, -1, self.node_id_dim),
        )
        edge_destinations = node_identifiers.gather(
            1,
            edge_index[:, 1].unsqueeze(-1).expand(-1, -1, self.node_id_dim),
        )
        edge_type = self.type_identifiers[1].view(1, 1, self.type_id_dim).expand(
            batch_size, edge_features.shape[1], self.type_id_dim
        )
        edge_tokens = torch.cat(
            (edge_features, edge_sources, edge_destinations, edge_type), dim=-1
        )
        return torch.cat((node_tokens, edge_tokens), dim=1)

    @staticmethod
    def _scaled_dot_product_attention(query, key, value):
        scale = math.sqrt(query.shape[-1])
        weights = torch.softmax(query @ key.transpose(-2, -1) / scale, dim=-1)
        return weights @ value, weights

    def _run_block(self, hidden, block):
        batch_size, sequence_length, _ = hidden.shape
        qkv = block["qkv"](block["norm1"](hidden))
        qkv = qkv.reshape(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attended, _ = self._scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(1, 2).reshape_as(hidden)
        hidden = hidden + block["attention_dropout"](
            block["attention_output"](attended)
        )
        mlp_output = block["mlp_out"](F.gelu(block["mlp_in"](block["norm2"](hidden))))
        return hidden + block["mlp_dropout"](mlp_output)

    def _encode(self, node_features, edge_features, edge_index, node_identifiers):
        tokens = self._augment_tokens(
            node_features, edge_features, edge_index, node_identifiers
        )
        projected = self.input_projection(tokens)
        graph_token = self.graph_token.expand(node_features.shape[0], -1, -1)
        hidden = torch.cat((graph_token, projected), dim=1)
        for block in self.blocks:
            hidden = self._run_block(hidden, block)
        return self.final_norm(hidden)

    def forward(self, node_features, edge_features, edge_index, node_identifiers):
        encoded = self._encode(
            node_features, edge_features, edge_index, node_identifiers
        )
        return self.output_projection(encoded[:, 0])
