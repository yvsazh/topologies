import torch
from torch import nn
from torch.nn import functional as F


class AWQLinear(nn.Module):
    component_id = "awq"

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        bits=4,
        group_size=128,
        grid_size=20,
        clip=True,
        clip_grid_size=20,
        max_shrink=0.5,
        scale_floor=1e-4,
    ):
        super().__init__()
        if not isinstance(in_features, int) or isinstance(in_features, bool) or in_features <= 0:
            raise ValueError("in_features must be a positive integer")
        if not isinstance(out_features, int) or isinstance(out_features, bool) or out_features <= 0:
            raise ValueError("out_features must be a positive integer")
        if type(bias) is not bool:
            raise ValueError("bias must be a boolean")
        if type(bits) is not int or bits not in (3, 4):
            raise ValueError("bits must be 3 or 4")
        if type(group_size) is not int:
            raise ValueError("group_size must be an integer")
        if group_size == -1:
            group_size = in_features
        if group_size <= 0 or in_features % group_size != 0:
            raise ValueError("group_size must be -1 or a positive divisor of in_features")
        if not isinstance(grid_size, int) or isinstance(grid_size, bool) or grid_size <= 0:
            raise ValueError("grid_size must be a positive integer")
        if type(clip) is not bool:
            raise ValueError("clip must be a boolean")
        if (
            not isinstance(clip_grid_size, int)
            or isinstance(clip_grid_size, bool)
            or clip_grid_size <= 0
        ):
            raise ValueError("clip_grid_size must be a positive integer")
        if (
            not isinstance(max_shrink, (int, float))
            or isinstance(max_shrink, bool)
            or not torch.isfinite(torch.tensor(float(max_shrink))).item()
            or max_shrink <= 0
            or max_shrink > 1
        ):
            raise ValueError("max_shrink must be finite and in the interval (0, 1]")
        if (
            not isinstance(scale_floor, (int, float))
            or isinstance(scale_floor, bool)
            or not torch.isfinite(torch.tensor(float(scale_floor))).item()
            or scale_floor <= 0
        ):
            raise ValueError("scale_floor must be a finite positive number")
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.num_groups = in_features // group_size
        self.grid_size = grid_size
        self.clip = clip
        self.clip_grid_size = clip_grid_size
        self.max_shrink = float(max_shrink)
        self.scale_floor = float(scale_floor)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.register_buffer("scale", torch.ones(in_features))
        self.register_buffer(
            "clip_bounds", torch.full((out_features, self.num_groups), float("inf"))
        )
        self.selected_alpha = 0.0
        self.calibration_mse = float("inf")
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            bound = self.in_features**-0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def _quantize_groups(self, weights):
        grouped = weights.reshape(self.out_features, self.num_groups, self.group_size)
        minimum = grouped.amin(dim=-1, keepdim=True)
        maximum = grouped.amax(dim=-1, keepdim=True)
        levels = (1 << self.bits) - 1
        step = (maximum - minimum).clamp_min(1e-5) / levels
        zero = torch.clamp(torch.round(-minimum / step), 0, levels)
        integers = torch.clamp(torch.round(grouped / step) + zero, 0, levels)
        return ((integers - zero) * step).reshape(self.out_features, self.in_features)

    def quantize_weight(self):
        scaled_weight = self.weight * self.scale
        if self.clip:
            bounds = self.clip_bounds.repeat_interleave(self.group_size, dim=1)
            scaled_weight = torch.maximum(torch.minimum(scaled_weight, bounds), -bounds)
        return self._quantize_groups(scaled_weight)

    def _search_clipping(self, calibration, scale):
        scaled_input = calibration / scale
        scaled_weight = self.weight * scale
        candidate_count = max(1, int(self.max_shrink * self.clip_grid_size))
        ratios = [1 - index / self.clip_grid_size for index in range(candidate_count)]
        selected = torch.empty_like(self.clip_bounds)
        for output_index in range(self.out_features):
            for group_index in range(self.num_groups):
                start = group_index * self.group_size
                end = start + self.group_size
                group_weight = scaled_weight[output_index, start:end]
                group_input = scaled_input[:, start:end]
                reference = group_input @ group_weight
                maximum = group_weight.abs().max()
                best_bound = maximum
                best_error = None
                for ratio in ratios:
                    bound = maximum * ratio
                    clipped = torch.clamp(group_weight, min=-bound, max=bound)
                    minimum = clipped.min()
                    maximum_value = clipped.max()
                    levels = (1 << self.bits) - 1
                    step = (maximum_value - minimum).clamp_min(1e-5) / levels
                    zero = torch.clamp(torch.round(-minimum / step), 0, levels)
                    integers = torch.clamp(torch.round(clipped / step) + zero, 0, levels)
                    quantized = (integers - zero) * step
                    estimate = group_input @ quantized
                    error = (estimate - reference).square().mean()
                    if best_error is None or error.item() < best_error:
                        best_error = error.item()
                        best_bound = bound
                selected[output_index, group_index] = best_bound
        self.clip_bounds.copy_(selected)

    def calibrate(self, features):
        if not isinstance(features, torch.Tensor) or features.ndim < 1:
            raise ValueError("calibration features must be a tensor with a feature dimension")
        if features.shape[-1] != self.in_features:
            raise ValueError("calibration feature width must equal in_features")
        if features.numel() == 0:
            raise ValueError("calibration features must not be empty")
        if not torch.is_floating_point(features):
            raise ValueError("calibration features must have a floating point dtype")
        calibration = features.detach().reshape(-1, self.in_features).to(
            device=self.weight.device, dtype=self.weight.dtype
        )
        if not torch.isfinite(calibration).all().item():
            raise ValueError("calibration features must contain only finite values")
        with torch.no_grad():
            reference = F.linear(calibration, self.weight, self.bias)
            activation = calibration.abs().mean(dim=0).clamp_min(self.scale_floor)
            best_loss = None
            best_scale = None
            best_alpha = 0.0
            for index in range(self.grid_size):
                alpha = index / self.grid_size
                candidate = activation.pow(alpha).clamp_min(self.scale_floor)
                candidate = candidate / torch.sqrt(candidate.max() * candidate.min())
                candidate_weight = self._quantize_groups(self.weight * candidate)
                estimate = F.linear(calibration / candidate, candidate_weight, self.bias)
                loss = (estimate - reference).square().mean()
                if best_loss is None or loss.item() < best_loss:
                    best_loss = loss.item()
                    best_scale = candidate
                    best_alpha = alpha
            self.scale.copy_(best_scale)
            self.selected_alpha = best_alpha
            self.calibration_mse = best_loss
            if self.clip:
                self._search_clipping(calibration, best_scale)
            else:
                self.clip_bounds.fill_(float("inf"))
        return self

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim < 1:
            raise ValueError("input must be a tensor with a feature dimension")
        if x.shape[-1] != self.in_features:
            raise ValueError("input feature width must equal in_features")
        scaled_weight = self.weight * self.scale
        if self.clip:
            bounds = self.clip_bounds.repeat_interleave(self.group_size, dim=1)
            scaled_weight = torch.maximum(torch.minimum(scaled_weight, bounds), -bounds)
        quantized_weight = self._quantize_groups(scaled_weight)
        weight_for_forward = scaled_weight + (quantized_weight - scaled_weight).detach()
        return F.linear(x / self.scale, weight_for_forward, self.bias)
