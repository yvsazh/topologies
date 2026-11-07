import math

import torch
from torch import nn
from torch.nn import functional as F


class GriffinBlock(nn.Module):
    component_id = "griffin-block"
    PRESETS = {
        "tiny": {"width": 12, "recurrent_width": 16, "num_gate_blocks": 16, "conv_width": 4},
        "small": {"width": 48, "recurrent_width": 64, "num_gate_blocks": 16, "conv_width": 4},
        "base": {"width": 768, "recurrent_width": 1024, "num_gate_blocks": 16, "conv_width": 4},
    }

    def __init__(
        self,
        width,
        recurrent_width=None,
        num_gate_blocks=16,
        conv_width=4,
        final_w_init_variance_scale=1.0,
    ):
        super().__init__()
        recurrent_width = width if recurrent_width is None else recurrent_width
        dimensions = (width, recurrent_width, num_gate_blocks, conv_width)
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in dimensions):
            raise ValueError("widths, gate block count, and convolution width must be positive integers")
        if recurrent_width % num_gate_blocks != 0:
            raise ValueError("recurrent_width must be divisible by num_gate_blocks")
        if (
            not isinstance(final_w_init_variance_scale, (int, float))
            or isinstance(final_w_init_variance_scale, bool)
            or not math.isfinite(final_w_init_variance_scale)
            or final_w_init_variance_scale <= 0
        ):
            raise ValueError("final_w_init_variance_scale must be a positive finite number")
        self.width = width
        self.recurrent_width = recurrent_width
        self.num_gate_blocks = num_gate_blocks
        self.conv_width = conv_width
        self.block_width = recurrent_width // num_gate_blocks
        self.input_projection = nn.Linear(width, recurrent_width)
        self.gelu_projection = nn.Linear(width, recurrent_width)
        self.conv_weight = nn.Parameter(torch.empty(conv_width, recurrent_width))
        self.conv_bias = nn.Parameter(torch.zeros(recurrent_width))
        self.recurrence_gate_weight = nn.Parameter(
            torch.empty(num_gate_blocks, self.block_width, self.block_width)
        )
        self.recurrence_gate_bias = nn.Parameter(torch.zeros(recurrent_width))
        self.input_gate_weight = nn.Parameter(
            torch.empty(num_gate_blocks, self.block_width, self.block_width)
        )
        self.input_gate_bias = nn.Parameter(torch.zeros(recurrent_width))
        self.output_projection = nn.Linear(recurrent_width, width)
        retention_power = torch.empty(recurrent_width).uniform_(0.9, 0.999)
        self.lambda_logit = nn.Parameter(torch.logit(retention_power.pow(1.0 / 8.0)))
        nn.init.uniform_(self.conv_weight, -math.sqrt(3.0 / conv_width), math.sqrt(3.0 / conv_width))
        gate_bound = math.sqrt(3.0 / self.block_width)
        nn.init.uniform_(self.recurrence_gate_weight, -gate_bound, gate_bound)
        nn.init.uniform_(self.input_gate_weight, -gate_bound, gate_bound)
        with torch.no_grad():
            self.output_projection.weight.mul_(math.sqrt(final_w_init_variance_scale))

    def _apply_gate(self, values, weight, bias):
        grouped = values.reshape(values.shape[0], values.shape[1], self.num_gate_blocks, self.block_width)
        projected = torch.einsum("btgi,goi->btgo", grouped, weight)
        return projected.reshape(values.shape[0], values.shape[1], self.recurrent_width) + bias

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError("input must have shape [batch, time, width]")
        if x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError("batch and time dimensions must be nonempty")
        if x.shape[-1] != self.width:
            raise ValueError("input width must match the configured width")
        projected = self.input_projection(x)
        convolved = self.conv_bias.view(1, 1, -1).expand_as(projected)
        for lag in range(min(self.conv_width, x.shape[1])):
            shifted = projected if lag == 0 else torch.cat(
                (projected.new_zeros(projected.shape[0], lag, self.recurrent_width), projected[:, :-lag]),
                dim=1,
            )
            convolved = convolved + shifted * self.conv_weight[lag].view(1, 1, -1)
        recurrence_gate = torch.sigmoid(
            self._apply_gate(convolved, self.recurrence_gate_weight, self.recurrence_gate_bias)
        )
        input_gate = torch.sigmoid(self._apply_gate(convolved, self.input_gate_weight, self.input_gate_bias))
        log_base_retention = F.logsigmoid(self.lambda_logit).view(1, -1)
        state = x.new_zeros((x.shape[0], self.recurrent_width))
        states = []
        for current, recurrence, input_value in zip(
            convolved.unbind(dim=1), recurrence_gate.unbind(dim=1), input_gate.unbind(dim=1)
        ):
            retention = torch.exp(8.0 * recurrence * log_base_retention)
            input_scale = torch.sqrt(torch.clamp(1.0 - retention.square(), min=0.0))
            state = retention * state + input_scale * (input_value * current)
            states.append(state)
        hidden = torch.stack(states, dim=1)
        gated = hidden * F.gelu(self.gelu_projection(x))
        return self.output_projection(gated)
