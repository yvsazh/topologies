import math

import torch
from torch import nn
from torch.nn import functional as F


class QALoRALinear(nn.Module):
    component_id = "qa-lora"
    PRESETS = {
        "tiny": {
            "in_features": 4,
            "out_features": 2,
            "rank": 1,
            "group_size": 2,
            "num_bits": 2,
            "scale": 1.0,
            "bias": False,
        },
        "small": {
            "in_features": 32,
            "out_features": 16,
            "rank": 4,
            "group_size": 8,
            "num_bits": 4,
            "scale": 1.0,
            "bias": False,
        },
    }

    def __init__(self, in_features, out_features, rank, scale, group_size=32, num_bits=4,
                 bias=False):
        super().__init__()
        for name, value in (("in_features", in_features), ("out_features", out_features),
                            ("rank", rank), ("group_size", group_size), ("num_bits", num_bits)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if num_bits > 53:
            raise ValueError("num_bits must not exceed 53 for exact float64 quantization")
        if in_features % group_size:
            raise ValueError("group_size must divide in_features")
        if type(scale) not in (int, float):
            raise ValueError("scale must be a finite number")
        try:
            scale = float(scale)
        except OverflowError as exc:
            raise ValueError("scale must be a finite number") from exc
        if not math.isfinite(scale):
            raise ValueError("scale must be a finite number")
        if type(bias) is not bool:
            raise ValueError("bias must be a boolean")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.group_size = group_size
        self.num_groups = in_features // group_size
        self.num_bits = num_bits
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features), requires_grad=False)
        else:
            self.register_parameter("bias", None)
        self.A = nn.Parameter(torch.empty(rank, self.num_groups))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        self.register_buffer(
            "quantization_codes",
            torch.empty(out_features, self.num_groups, group_size, dtype=torch.int64),
        )
        self.register_buffer(
            "quantization_scale",
            torch.empty(out_features, self.num_groups, 1),
        )
        self.register_buffer(
            "quantization_offset",
            torch.empty(out_features, self.num_groups, 1),
        )
        self.is_merged = False
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(in_features)
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.xavier_uniform_(self.A)
        self._refresh_quantization()

    @torch.no_grad()
    def _refresh_quantization(self):
        if self.is_merged:
            return
        grouped_weight = self.weight.reshape(
            self.out_features, self.num_groups, self.group_size
        )
        if self.num_bits > 24:
            grouped_weight = grouped_weight.to(torch.float64)
        elif grouped_weight.dtype in (torch.float16, torch.bfloat16):
            grouped_weight = grouped_weight.to(torch.float32)
        minimum = grouped_weight.amin(dim=-1, keepdim=True)
        maximum = grouped_weight.amax(dim=-1, keepdim=True)
        qmax = 2**self.num_bits - 1
        quantization_scale = (maximum - minimum) / qmax
        quantization_scale = torch.where(
            maximum == minimum, torch.ones_like(quantization_scale), quantization_scale
        )
        codes = torch.round((grouped_weight - minimum) / quantization_scale)
        codes = codes.clamp(0, qmax).to(torch.int64)
        self.quantization_codes = codes
        self.quantization_scale = quantization_scale
        self.quantization_offset = minimum

    def dequantized_weight(self):
        self._refresh_quantization()
        weight = (
            self.quantization_scale * self.quantization_codes.to(self.quantization_scale.dtype)
            + self.quantization_offset
        ).to(self.weight.dtype)
        return weight.reshape(self.out_features, self.in_features)

    def merge_adapter(self):
        if self.is_merged:
            return self
        self._refresh_quantization()
        shift = self.scale * (self.B @ self.A).reshape(
            self.out_features, self.num_groups, 1
        )
        with torch.no_grad():
            self.quantization_offset.add_(shift)
        self.is_merged = True
        return self

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim == 0 or not x.is_floating_point():
            raise ValueError("x must be a floating-point tensor with a final feature dimension")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"x must have a final dimension of {self.in_features}")
        if x.device != self.weight.device or x.dtype != self.weight.dtype:
            raise ValueError("x must match the device and dtype of the layer weights")
        base = F.linear(x, self.dequantized_weight(), self.bias)
        if self.is_merged:
            return base
        grouped_input = x.reshape(*x.shape[:-1], self.num_groups, self.group_size)
        summed_input = grouped_input.sum(dim=-1)
        update = F.linear(F.linear(summed_input, self.A), self.B)
        return base + self.scale * update
