import math

import torch
from torch import nn


class ClimaX(nn.Module):
    architecture_id = "climax"
    PRESETS = {
        "tiny": {
            "num_variables": 2,
            "img_size": (8, 8),
            "patch_size": 4,
            "embed_dim": 32,
            "depth": 2,
            "num_heads": 4,
            "mlp_ratio": 4.0,
            "decoder_depth": 1,
            "drop_rate": 0.0,
            "drop_path": 0.0,
        },
        "small": {
            "num_variables": 2,
            "img_size": (32, 32),
            "patch_size": 4,
            "embed_dim": 256,
            "depth": 8,
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "decoder_depth": 2,
            "drop_rate": 0.1,
            "drop_path": 0.1,
        },
    }

    def __init__(
        self,
        num_variables=2,
        img_size=(8, 8),
        patch_size=4,
        embed_dim=32,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        decoder_depth=1,
        drop_rate=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        if not isinstance(num_variables, int) or isinstance(num_variables, bool) or num_variables < 1:
            raise ValueError("num_variables must be a positive integer")
        if isinstance(img_size, int) and not isinstance(img_size, bool):
            img_size = (img_size, img_size)
        if not isinstance(img_size, (tuple, list)) or len(img_size) != 2:
            raise ValueError("img_size must be an integer or a pair of positive integers")
        if any(not isinstance(size, int) or isinstance(size, bool) or size < 1 for size in img_size):
            raise ValueError("img_size must contain positive integers")
        if not isinstance(patch_size, int) or isinstance(patch_size, bool) or patch_size < 1:
            raise ValueError("patch_size must be a positive integer")
        if any(size % patch_size for size in img_size):
            raise ValueError("img_size dimensions must be divisible by patch_size")
        if not isinstance(embed_dim, int) or isinstance(embed_dim, bool) or embed_dim < 1:
            raise ValueError("embed_dim must be a positive integer")
        if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1:
            raise ValueError("depth must be a positive integer")
        if not isinstance(num_heads, int) or isinstance(num_heads, bool) or num_heads < 1:
            raise ValueError("num_heads must be a positive integer")
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        invalid_mlp_ratio = not isinstance(mlp_ratio, (int, float)) or isinstance(mlp_ratio, bool)
        if not invalid_mlp_ratio:
            invalid_mlp_ratio = (
                not math.isfinite(mlp_ratio)
                or mlp_ratio <= 0
                or int(embed_dim * mlp_ratio) < 1
            )
        if invalid_mlp_ratio:
            raise ValueError("mlp_ratio must produce a positive finite hidden dimension")
        if not isinstance(decoder_depth, int) or isinstance(decoder_depth, bool) or decoder_depth < 0:
            raise ValueError("decoder_depth must be a non-negative integer")
        invalid_drop_rate = not isinstance(drop_rate, (int, float)) or isinstance(drop_rate, bool)
        if not invalid_drop_rate:
            invalid_drop_rate = not math.isfinite(drop_rate) or not 0 <= drop_rate < 1
        if invalid_drop_rate:
            raise ValueError("drop_rate must be a finite number in [0, 1)")
        invalid_drop_path = not isinstance(drop_path, (int, float)) or isinstance(drop_path, bool)
        if not invalid_drop_path:
            invalid_drop_path = not math.isfinite(drop_path) or not 0 <= drop_path < 1
        if invalid_drop_path:
            raise ValueError("drop_path must be a finite number in [0, 1)")

        self.num_variables = num_variables
        self.img_size = tuple(img_size)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.grid_size = tuple(size // patch_size for size in self.img_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_embeddings = nn.ModuleList(
            nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
            for _ in range(num_variables)
        )
        self.variable_embeddings = nn.Parameter(torch.zeros(num_variables, embed_dim))
        self.aggregation_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.variable_aggregation = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.positional_embedding = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.lead_time_embedding = nn.Linear(1, embed_dim)

        path_rates = torch.linspace(0, drop_path, depth).tolist()
        self.drop_path_rates = tuple(path_rates)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.blocks = nn.ModuleList()
        for _ in path_rates:
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "norm1": nn.LayerNorm(embed_dim),
                        "attention": nn.MultiheadAttention(
                            embed_dim, num_heads, dropout=drop_rate, batch_first=True
                        ),
                        "attention_dropout": nn.Dropout(drop_rate),
                        "norm2": nn.LayerNorm(embed_dim),
                        "mlp_input": nn.Linear(embed_dim, hidden_dim),
                        "mlp_activation": nn.GELU(),
                        "mlp_dropout1": nn.Dropout(drop_rate),
                        "mlp_output": nn.Linear(hidden_dim, embed_dim),
                        "mlp_dropout2": nn.Dropout(drop_rate),
                    }
                )
            )
        self.norm = nn.LayerNorm(embed_dim)
        decoder_layers = []
        for _ in range(decoder_depth):
            decoder_layers.extend((nn.Linear(embed_dim, embed_dim), nn.GELU()))
        decoder_layers.append(nn.Linear(embed_dim, num_variables * patch_size * patch_size))
        self.prediction_head = nn.Sequential(*decoder_layers)

    def _normalize_ids(self, ids, name, device):
        try:
            ids = torch.as_tensor(ids, device=device)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError(f"{name} must be a one-dimensional sequence of integer IDs") from error
        if ids.ndim != 1 or ids.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError(f"{name} must be a one-dimensional sequence of integer IDs")
        ids = ids.to(dtype=torch.long)
        if ids.numel() == 0:
            raise ValueError(f"{name} must contain at least one variable ID")
        if torch.unique(ids).numel() != ids.numel():
            raise ValueError(f"{name} must contain unique variable IDs")
        if torch.any(ids < 0) or torch.any(ids >= self.num_variables):
            raise ValueError(f"{name} IDs must be in range [0, {self.num_variables})")
        return ids

    def aggregate_variables(self, tokens):
        batch, variables, locations, width = tokens.shape
        by_location = tokens.permute(0, 2, 1, 3).reshape(batch * locations, variables, width)
        query = self.aggregation_query.expand(batch * locations, -1, -1)
        aggregated = self.variable_aggregation(query, by_location, by_location, need_weights=False)[0]
        return aggregated.reshape(batch, locations, width)

    def unpatchify(self, patches):
        batch, locations, values = patches.shape
        height, width = self.grid_size
        patch_size = self.patch_size
        expected_locations = height * width
        expected_values = self.num_variables * patch_size * patch_size
        if locations != expected_locations or values != expected_values:
            raise ValueError("patch predictions do not match the configured grid and variables")
        patches = patches.reshape(batch, height, width, patch_size, patch_size, self.num_variables)
        return patches.permute(0, 5, 1, 3, 2, 4).reshape(
            batch, self.num_variables, self.img_size[0], self.img_size[1]
        )

    def _drop_path(self, x, rate):
        if not self.training or rate == 0:
            return x
        keep_probability = 1 - rate
        mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(mask_shape).bernoulli_(keep_probability)
        return x * mask / keep_probability

    def _transformer_block(self, x, block, drop_path_rate):
        normalized = block["norm1"](x)
        attended = block["attention"](normalized, normalized, normalized, need_weights=False)[0]
        x = x + self._drop_path(block["attention_dropout"](attended), drop_path_rate)
        normalized = block["norm2"](x)
        transformed = block["mlp_input"](normalized)
        transformed = block["mlp_activation"](transformed)
        transformed = block["mlp_dropout1"](transformed)
        transformed = block["mlp_output"](transformed)
        transformed = block["mlp_dropout2"](transformed)
        return x + self._drop_path(transformed, drop_path_rate)

    def forward(self, x, variable_ids, lead_time, output_variable_ids=None):
        if not isinstance(x, torch.Tensor) or x.ndim != 4:
            raise ValueError("x must be a rank 4 tensor with shape [batch, variables, height, width]")
        batch, input_variables, height, width = x.shape
        if (height, width) != self.img_size:
            raise ValueError(f"x spatial dimensions must match img_size {self.img_size}")
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("x spatial dimensions must be divisible by patch_size")
        variable_ids = self._normalize_ids(variable_ids, "variable_ids", x.device)
        if variable_ids.numel() != input_variables:
            raise ValueError("variable_ids length must match the input channel count")
        if output_variable_ids is None:
            output_variable_ids = torch.arange(self.num_variables, device=x.device)
        else:
            output_variable_ids = self._normalize_ids(
                output_variable_ids, "output_variable_ids", x.device
            )
        if not isinstance(lead_time, torch.Tensor):
            try:
                lead_time = torch.as_tensor(lead_time)
            except (TypeError, ValueError, RuntimeError) as error:
                raise ValueError("lead_time must have shape [batch]") from error
        if lead_time.ndim != 1 or lead_time.shape[0] != batch:
            raise ValueError("lead_time must have shape [batch]")
        lead_time = lead_time.to(device=x.device, dtype=x.dtype).reshape(batch, 1)

        tokens = []
        for input_index, variable_id in enumerate(variable_ids.tolist()):
            variable_map = x[:, input_index : input_index + 1]
            embedded = self.patch_embeddings[variable_id](variable_map)
            embedded = embedded.flatten(2).transpose(1, 2)
            tokens.append(embedded + self.variable_embeddings[variable_id].view(1, 1, -1))
        variable_tokens = torch.stack(tokens, dim=1)
        tokens = self.aggregate_variables(variable_tokens)
        tokens = tokens + self.positional_embedding + self.lead_time_embedding(lead_time).unsqueeze(1)
        for block, drop_path_rate in zip(self.blocks, self.drop_path_rates):
            tokens = self._transformer_block(tokens, block, drop_path_rate)
        tokens = self.norm(tokens)
        patches = self.prediction_head(tokens)
        predictions = self.unpatchify(patches)
        return predictions.index_select(1, output_variable_ids)
