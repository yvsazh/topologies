import math

import torch
from torch import nn


class RandomErasing(nn.Module):
    component_id = "random-erasing"

    def __init__(self, p=0.5, sl=0.02, sh=0.4, r1=0.3, r2=None,
                 fill_mode="random", fill_value=0.0):
        super().__init__()
        if not isinstance(p, (int, float)) or isinstance(p, bool) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("p must be a finite number in [0, 1]")
        if (
            not isinstance(sl, (int, float))
            or isinstance(sl, bool)
            or not math.isfinite(sl)
            or not isinstance(sh, (int, float))
            or isinstance(sh, bool)
            or not math.isfinite(sh)
            or not 0 < sl <= sh <= 1
        ):
            raise ValueError("area bounds must be finite numbers with 0 < sl <= sh <= 1")
        if r2 is None:
            r2 = 1.0 / r1 if isinstance(r1, (int, float)) and r1 != 0 else None
        if (
            not isinstance(r1, (int, float))
            or isinstance(r1, bool)
            or not math.isfinite(r1)
            or r1 <= 0
            or not isinstance(r2, (int, float))
            or isinstance(r2, bool)
            or not math.isfinite(r2)
            or r2 < r1
        ):
            raise ValueError("aspect bounds must be finite numbers with 0 < r1 <= r2")
        if fill_mode not in ("random", "constant"):
            raise ValueError("fill_mode must be 'random' or 'constant'")
        if (
            not isinstance(fill_value, (int, float))
            or isinstance(fill_value, bool)
            or not math.isfinite(fill_value)
        ):
            raise ValueError("fill_value must be a finite number")
        self.p = float(p)
        self.sl = float(sl)
        self.sh = float(sh)
        self.r1 = float(r1)
        self.r2 = float(r2)
        self.fill_mode = fill_mode
        self.fill_value = float(fill_value)

    def forward(self, x):
        if not torch.is_tensor(x) or x.ndim != 4:
            raise ValueError("input must be a tensor with NCHW shape")
        if any(size <= 0 for size in x.shape):
            raise ValueError("input dimensions must be positive")
        if not (x.is_floating_point() or x.dtype in (torch.uint8, torch.int16, torch.int32, torch.int64)):
            raise ValueError("input dtype must support real values in the fill range [0, 255]")
        if not self.training or self.p == 0:
            return x

        batch_size, channels, height, width = x.shape
        output = x.clone()
        for batch_index in range(batch_size):
            if torch.rand((), device=x.device).item() >= self.p:
                continue
            for _ in range(1000):
                area_ratio = self.sl + torch.rand((), device=x.device).item() * (self.sh - self.sl)
                aspect_ratio = self.r1 + torch.rand((), device=x.device).item() * (self.r2 - self.r1)
                target_area = area_ratio * height * width
                erase_height = max(1, int(math.sqrt(target_area * aspect_ratio)))
                erase_width = max(1, int(math.sqrt(target_area / aspect_ratio)))
                if erase_height <= height and erase_width <= width:
                    top = torch.randint(height - erase_height + 1, (), device=x.device).item()
                    left = torch.randint(width - erase_width + 1, (), device=x.device).item()
                    break
            else:
                raise ValueError("area and aspect bounds could not produce a rectangle that fits the input")

            region = (batch_index, slice(None), slice(top, top + erase_height),
                      slice(left, left + erase_width))
            if self.fill_mode == "constant":
                output[region] = self.fill_value
            elif x.is_floating_point():
                output[region] = torch.rand(
                    (channels, erase_height, erase_width), device=x.device, dtype=x.dtype
                ) * 255
            else:
                values = torch.rand((channels, erase_height, erase_width), device=x.device) * 256
                output[region] = values.floor().to(dtype=x.dtype)
        return output
