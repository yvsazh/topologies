import math

import torch
from torch import nn
from torch.nn import functional as F


class Perceiver(nn.Module):
    architecture_id = "perceiver"
    PRESETS = {
        "tiny": {
            "input_dim": 8,
            "num_classes": 2,
            "num_latents": 8,
            "latent_dim": 32,
            "depth": 1,
            "num_cross_attends": 1,
            "num_heads": 4,
            "cross_heads": 1,
            "mlp_ratio": 1,
            "dropout": 0.0,
            "latent_init_std": 0.02,
        },
        "small": {
            "input_dim": 32,
            "num_classes": 10,
            "num_latents": 32,
            "latent_dim": 64,
            "depth": 4,
            "num_cross_attends": 2,
            "num_heads": 8,
            "cross_heads": 1,
            "mlp_ratio": 1,
            "dropout": 0.0,
            "latent_init_std": 0.02,
        },
    }

    def __init__(
        self,
        input_dim,
        num_classes,
        num_latents=512,
        latent_dim=1024,
        depth=6,
        num_cross_attends=1,
        num_heads=8,
        cross_heads=1,
        mlp_ratio=1,
        dropout=0.0,
        latent_init_std=0.02,
    ):
        super().__init__()
        integer_values = {
            "input_dim": input_dim,
            "num_classes": num_classes,
            "num_latents": num_latents,
            "latent_dim": latent_dim,
            "depth": depth,
            "num_cross_attends": num_cross_attends,
            "num_heads": num_heads,
            "cross_heads": cross_heads,
            "mlp_ratio": mlp_ratio,
        }
        for name, value in integer_values.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if latent_dim % num_heads != 0:
            raise ValueError("latent_dim must be divisible by num_heads")
        if latent_dim % cross_heads != 0:
            raise ValueError("latent_dim must be divisible by cross_heads")
        if (
            not isinstance(dropout, (int, float))
            or isinstance(dropout, bool)
            or not math.isfinite(dropout)
            or not 0 <= dropout < 1
        ):
            raise ValueError("dropout must be a finite number in [0, 1)")
        if (
            not isinstance(latent_init_std, (int, float))
            or isinstance(latent_init_std, bool)
            or not math.isfinite(latent_init_std)
            or latent_init_std <= 0
        ):
            raise ValueError("latent_init_std must be a finite positive number")

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        self.depth = depth
        self.num_cross_attends = num_cross_attends
        self.num_heads = num_heads
        self.cross_heads = cross_heads
        self.mlp_ratio = mlp_ratio
        self.latents = nn.Parameter(torch.empty(1, num_latents, latent_dim))
        nn.init.trunc_normal_(
            self.latents,
            std=latent_init_std,
            a=-2 * latent_init_std,
            b=2 * latent_init_std,
        )

        self.cross_latent_norms = nn.ModuleList(nn.LayerNorm(latent_dim) for _ in range(num_cross_attends))
        self.cross_input_norms = nn.ModuleList(nn.LayerNorm(input_dim) for _ in range(num_cross_attends))
        self.cross_queries = nn.ModuleList(nn.Linear(latent_dim, latent_dim) for _ in range(num_cross_attends))
        self.cross_keys = nn.ModuleList(nn.Linear(input_dim, latent_dim) for _ in range(num_cross_attends))
        self.cross_values = nn.ModuleList(nn.Linear(input_dim, latent_dim) for _ in range(num_cross_attends))
        self.cross_outputs = nn.ModuleList(nn.Linear(latent_dim, latent_dim) for _ in range(num_cross_attends))
        self.cross_dropouts = nn.ModuleList(nn.Dropout(dropout) for _ in range(num_cross_attends))

        self.transformer_blocks = nn.ModuleList(
            nn.ModuleDict(
                {
                    "attention_norm": nn.LayerNorm(latent_dim),
                    "query": nn.Linear(latent_dim, latent_dim),
                    "key": nn.Linear(latent_dim, latent_dim),
                    "value": nn.Linear(latent_dim, latent_dim),
                    "attention_output": nn.Linear(latent_dim, latent_dim),
                    "attention_dropout": nn.Dropout(dropout),
                    "mlp_norm": nn.LayerNorm(latent_dim),
                    "mlp_in": nn.Linear(latent_dim, latent_dim * mlp_ratio),
                    "mlp_out": nn.Linear(latent_dim * mlp_ratio, latent_dim),
                    "mlp_dropout": nn.Dropout(dropout),
                }
            )
            for _ in range(num_cross_attends * depth)
        )
        self.classifier = nn.Linear(latent_dim, num_classes)

    @staticmethod
    def _attention(query, key, value):
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError("query, key, and value must have shape [batch, heads, sequence, head_dim]")
        if query.shape[:2] != key.shape[:2] or key.shape[:2] != value.shape[:2]:
            raise ValueError("query, key, and value must have matching batch and head dimensions")
        if query.shape[-1] != key.shape[-1] or key.shape[-2:] != value.shape[-2:]:
            raise ValueError("query, key, and value dimensions are incompatible")
        if key.shape[-2] == 0:
            raise ValueError("attention key and value sequences must be nonempty")
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
        return torch.matmul(torch.softmax(scores, dim=-1), value)

    def _split_heads(self, tensor, heads):
        batch, length, _ = tensor.shape
        return tensor.reshape(batch, length, heads, self.latent_dim // heads).transpose(1, 2)

    def _encode(self, inputs):
        batch = inputs.shape[0]
        latents = self.latents.expand(batch, -1, -1)
        for stage in range(self.num_cross_attends):
            normalized_latents = self.cross_latent_norms[stage](latents)
            normalized_inputs = self.cross_input_norms[stage](inputs)
            query = self._split_heads(self.cross_queries[stage](normalized_latents), self.cross_heads)
            key = self._split_heads(self.cross_keys[stage](normalized_inputs), self.cross_heads)
            value = self._split_heads(self.cross_values[stage](normalized_inputs), self.cross_heads)
            attended = self._attention(query, key, value)
            attended = attended.transpose(1, 2).reshape(batch, self.num_latents, self.latent_dim)
            latents = latents + self.cross_dropouts[stage](self.cross_outputs[stage](attended))

            start = stage * self.depth
            for block in self.transformer_blocks[start : start + self.depth]:
                normalized = block["attention_norm"](latents)
                query = self._split_heads(block["query"](normalized), self.num_heads)
                key = self._split_heads(block["key"](normalized), self.num_heads)
                value = self._split_heads(block["value"](normalized), self.num_heads)
                attended = self._attention(query, key, value)
                attended = attended.transpose(1, 2).reshape(batch, self.num_latents, self.latent_dim)
                latents = latents + block["attention_dropout"](block["attention_output"](attended))
                hidden = F.gelu(block["mlp_in"](block["mlp_norm"](latents)))
                latents = latents + block["mlp_dropout"](block["mlp_out"](hidden))
        return latents

    def forward(self, inputs):
        if not torch.is_tensor(inputs) or inputs.ndim != 3:
            raise ValueError("inputs must have shape [batch, sequence, input_dim]")
        if inputs.shape[0] <= 0:
            raise ValueError("input batch must be nonempty")
        if inputs.shape[1] <= 0:
            raise ValueError("input sequence must be nonempty")
        if inputs.shape[2] != self.input_dim:
            raise ValueError(f"inputs must have feature dimension {self.input_dim}")
        latents = self._encode(inputs)
        return self.classifier(latents.mean(dim=1))
