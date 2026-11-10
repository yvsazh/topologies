import math

import torch
from torch import nn
from torch.nn import functional as F


def _raise_value_error(message):
    raise ValueError(message)


class MuPLinear(nn.Module):
    component_id = "mup"
    PRESETS = {
        "tiny": {
            "in_features": 4,
            "out_features": 4,
            "role": "hidden",
            "base_in_features": 4,
            "base_out_features": 4,
        },
        "small": {
            "in_features": 16,
            "out_features": 16,
            "role": "hidden",
            "base_in_features": 4,
            "base_out_features": 4,
        },
    }

    def __init__(
        self,
        in_features,
        out_features,
        role="hidden",
        base_in_features=None,
        base_out_features=None,
        init_std=1.0,
        bias=False,
    ):
        super().__init__()
        for name, value in (("in_features", in_features), ("out_features", out_features)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                _raise_value_error(f"{name} must be a positive integer")
        if role not in ("input", "hidden", "output"):
            _raise_value_error("role must be 'input', 'hidden', or 'output'")
        if base_in_features is None:
            base_in_features = in_features
        if base_out_features is None:
            base_out_features = out_features
        for name, value in (
            ("base_in_features", base_in_features),
            ("base_out_features", base_out_features),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                _raise_value_error(f"{name} must be a positive integer")
        if isinstance(init_std, bool) or not isinstance(init_std, (int, float)):
            _raise_value_error("init_std must be a positive finite number")
        try:
            init_std = float(init_std)
        except OverflowError as error:
            raise ValueError("init_std must be a positive finite number") from error
        if not math.isfinite(init_std) or init_std <= 0:
            _raise_value_error("init_std must be a positive finite number")
        if not isinstance(bias, bool):
            _raise_value_error("bias must be a boolean")

        self.in_features = in_features
        self.out_features = out_features
        self.role = role
        self.base_in_features = base_in_features
        self.base_out_features = base_out_features
        self.init_std = init_std
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        fan_in = base_in_features if role == "output" else in_features
        nn.init.normal_(self.weight, mean=0.0, std=init_std / math.sqrt(fan_in))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim == 0 or not x.is_floating_point():
            raise ValueError("x must be a floating-point tensor with a final feature dimension")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"x must have a final dimension of {self.in_features}")
        output = F.linear(x, self.weight)
        if self.role == "output":
            output = output * (self.base_in_features / self.in_features)
        if self.bias is not None:
            output = output + self.bias
        return output

    def optimizer_groups(self, learning_rate, optimizer="adam"):
        if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
            _raise_value_error("learning_rate must be a positive finite number")
        try:
            learning_rate = float(learning_rate)
        except OverflowError as error:
            raise ValueError("learning_rate must be a positive finite number") from error
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            _raise_value_error("learning_rate must be a positive finite number")
        if not isinstance(optimizer, str):
            _raise_value_error("optimizer must be sgd, adam, adamw, adagrad, or rmsprop")
        optimizer = optimizer.lower()
        if optimizer not in ("sgd", "adam", "adamw", "adagrad", "rmsprop"):
            _raise_value_error("optimizer must be sgd, adam, adamw, adagrad, or rmsprop")

        if optimizer == "sgd":
            if self.role == "input":
                weight_scale = self.out_features / self.base_out_features
            elif self.role == "output":
                weight_scale = self.in_features / self.base_in_features
            else:
                weight_scale = 1.0
            bias_scale = self.out_features / self.base_out_features
        else:
            if self.role == "hidden":
                weight_scale = self.base_in_features / self.in_features
            else:
                weight_scale = 1.0
            bias_scale = 1.0

        groups = [{"params": [self.weight], "lr": learning_rate * weight_scale}]
        if self.bias is not None:
            groups.append({"params": [self.bias], "lr": learning_rate * bias_scale})
        return groups
