import torch
from torch import nn

class TabularMLP(nn.Module):
    architecture_id = 'tabular-mlp'

    @staticmethod
    def _is_integer(value: object) -> bool:
        return type(value) is int

    def __init__(self, numeric_features: int=1, categorical_cardinalities: tuple[int, ...] | list[int]=(), out_features: int=1, embedding_dims: tuple[int, ...] | list[int] | int=8, hidden_dims: tuple[int, ...] | list[int]=(128, 64), activation: str='relu', dropout: float=0.0, missing_numeric: str='error', *, norm: str='none') -> None:
        super().__init__()
        if not self._is_integer(numeric_features) or not self._is_integer(out_features):
            raise ValueError('numeric_features and out_features must be integers (bool is not allowed)')
        if not isinstance(categorical_cardinalities, (tuple, list)):
            raise ValueError('categorical_cardinalities must be a list or tuple of integers')
        cards = tuple(categorical_cardinalities)
        if self._is_integer(embedding_dims):
            widths = (embedding_dims,) * len(cards)
        elif isinstance(embedding_dims, (tuple, list)):
            widths = tuple(embedding_dims)
        else:
            raise ValueError('embedding_dims must be an integer or a list/tuple of integers')
        if self._is_integer(hidden_dims):
            hidden = (hidden_dims,)
        elif isinstance(hidden_dims, (tuple, list)):
            hidden = tuple(hidden_dims)
        else:
            raise ValueError('hidden_dims must be an integer or a list/tuple of integers')
        if numeric_features < 0 or out_features <= 0:
            raise ValueError('numeric_features must be >= 0 and out_features positive')
        if any((not self._is_integer(n) or n <= 0 for n in cards)):
            raise ValueError('categorical cardinalities must be positive integers')
        if len(widths) != len(cards) or any((not self._is_integer(n) or n <= 0 for n in widths)):
            raise ValueError('embedding_dims must provide one positive width per categorical field')
        if numeric_features + len(cards) == 0:
            raise ValueError('at least one input field is required')
        if not hidden or any((not self._is_integer(n) or n <= 0 for n in hidden)):
            raise ValueError('hidden_dims must be a nonempty sequence of positive integers')
        if activation not in {'relu', 'gelu', 'silu', 'tanh'}:
            raise ValueError('activation must be relu, gelu, silu, or tanh')
        if norm not in {'none', 'batch', 'layer'}:
            raise ValueError('norm must be none, batch, or layer')
        if not 0.0 <= dropout < 1.0:
            raise ValueError('dropout must be in [0, 1)')
        if missing_numeric not in {'error', 'zero'}:
            raise ValueError('missing_numeric must be error or zero')
        self.numeric_features = numeric_features
        self.categorical_cardinalities = cards
        self.missing_numeric = missing_numeric
        self.embeddings = nn.ModuleList([nn.Embedding(card + 1, dim) for card, dim in zip(cards, widths)])
        width = numeric_features + sum(widths)
        act = {'relu': nn.ReLU, 'gelu': nn.GELU, 'silu': nn.SiLU, 'tanh': nn.Tanh}[activation]
        modules: list[nn.Module] = []
        for hidden_width in hidden:
            modules.append(nn.Linear(width, hidden_width))
            if norm == 'batch':
                modules.append(nn.BatchNorm1d(hidden_width))
            elif norm == 'layer':
                modules.append(nn.LayerNorm(hidden_width))
            modules.extend((act(), nn.Dropout(dropout)))
            width = hidden_width
        modules.append(nn.Linear(width, out_features))
        self.network = nn.Sequential(*modules)

    def forward(self, features: torch.Tensor, categorical: torch.Tensor | None=None) -> torch.Tensor:
        numeric = features
        if not isinstance(numeric, torch.Tensor):
            raise ValueError('numeric features must be a torch.Tensor')
        if numeric.ndim != 2 or numeric.shape[1] != self.numeric_features:
            raise ValueError(f'expected numeric input [B, {self.numeric_features}]')
        if numeric.shape[0] <= 0:
            raise ValueError('numeric input batch size must be positive')
        if not numeric.is_floating_point():
            raise ValueError('numeric features must have a floating-point dtype')
        if self.numeric_features:
            numeric = numeric.to(dtype=self.network[0].weight.dtype)
            missing = torch.isnan(numeric)
            if numeric.device.type != 'meta' and bool(missing.any()) and (self.missing_numeric == 'error'):
                raise ValueError('numeric input contains missing (NaN) values')
            if numeric.device.type != 'meta' and bool(torch.isinf(numeric).any()):
                raise ValueError('numeric input contains positive or negative infinity; preprocess infinities explicitly')
            numeric = torch.nan_to_num(numeric, nan=0.0)
            parts = [numeric]
        else:
            parts = []
        if self.embeddings:
            if categorical is None or not isinstance(categorical, torch.Tensor):
                raise ValueError(f'expected categorical input [B, {len(self.embeddings)}] as a torch.Tensor')
            if categorical.ndim != 2 or categorical.shape != (numeric.shape[0], len(self.embeddings)):
                raise ValueError(f'expected categorical input [B, {len(self.embeddings)}]')
            if categorical.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                raise ValueError('categorical IDs must have an integer dtype (bool/floating values are not allowed)')
            if categorical.device != numeric.device:
                raise ValueError('numeric and categorical tensors must be on the same device')
            categorical = categorical.to(dtype=torch.long)
            for index, (cardinality, embedding) in enumerate(zip(self.categorical_cardinalities, self.embeddings)):
                ids = categorical[:, index]
                if ids.device.type != 'meta' and bool(((ids < -1) | (ids >= cardinality)).any()):
                    raise ValueError(f'categorical field {index} IDs must be -1 (missing) or in [0, {cardinality})')
                parts.append(embedding(torch.where(ids == -1, cardinality, ids)))
        elif categorical is not None and (not isinstance(categorical, torch.Tensor) or categorical.shape != (numeric.shape[0], 0)):
            raise ValueError('this model declares no categorical fields')
        return self.network(torch.cat(parts, dim=1))
