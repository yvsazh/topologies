import torch
from torch import nn

class MLP(nn.Module):
    architecture_id = 'mlp'

    def __init__(self, in_features: int=1, out_features: int=1, hidden_dims: tuple[int, ...] | list[int]=(128, 64), activation: str='relu', norm: str='none', dropout: float=0.0) -> None:
        super().__init__()
        if type(in_features) is not int or type(out_features) is not int:
            raise ValueError('in_features and out_features must be integers (bool is not allowed)')
        if type(hidden_dims) is int:
            dims = (hidden_dims,)
        elif isinstance(hidden_dims, (tuple, list)):
            dims = tuple(hidden_dims)
        else:
            raise ValueError('hidden_dims must be an integer or a list/tuple of integers')
        if in_features <= 0 or out_features <= 0:
            raise ValueError('in_features and out_features must be positive')
        if not dims or any((type(width) is not int or width <= 0 for width in dims)):
            raise ValueError('hidden_dims must be a nonempty sequence of positive integers')
        if activation not in {'relu', 'gelu', 'silu', 'tanh'}:
            raise ValueError('activation must be relu, gelu, silu, or tanh')
        if norm not in {'none', 'batch', 'layer'}:
            raise ValueError('norm must be none, batch, or layer')
        if not 0.0 <= dropout < 1.0:
            raise ValueError('dropout must be in [0, 1)')
        self.in_features, self.out_features = (in_features, out_features)
        self.hidden_dims = dims
        act = {'relu': nn.ReLU, 'gelu': nn.GELU, 'silu': nn.SiLU, 'tanh': nn.Tanh}[activation]
        modules: list[nn.Module] = []
        width = in_features
        for hidden in dims:
            modules.append(nn.Linear(width, hidden))
            if norm == 'batch':
                modules.append(nn.BatchNorm1d(hidden))
            elif norm == 'layer':
                modules.append(nn.LayerNorm(hidden))
            modules.extend((act(), nn.Dropout(dropout)))
            width = hidden
        modules.append(nn.Linear(width, out_features))
        self.network = nn.Sequential(*modules)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not isinstance(inputs, torch.Tensor):
            raise ValueError('inputs must be a torch.Tensor')
        if inputs.ndim != 2 or inputs.shape[1] != self.in_features:
            raise ValueError(f'expected input [B, {self.in_features}], got {tuple(inputs.shape)}')
        if not inputs.is_floating_point():
            raise ValueError('inputs must have a floating-point dtype')
        return self.network(inputs)
