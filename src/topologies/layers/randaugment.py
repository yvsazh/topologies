

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RandAugment(nn.Module):
    component_id = "randaugment"
    """RandAugment with a fixed operation count and magnitude.

    Inputs are NCHW floating tensors in [0, 1]. All randomness is drawn from
    the caller-provided ``torch.Generator`` (including policy selection).
    """

    OPS = ("identity", "autocontrast", "equalize", "rotate", "solarize", "color",
           "posterize", "contrast", "brightness", "sharpness", "shear-x", "shear-y",
           "translate-x", "translate-y")


    def __init__(self, n: int = 2, m: float = 9):
        super().__init__()
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError("n must be a non-negative integer")
        if isinstance(m, bool) or not isinstance(m, (int, float)) or not math.isfinite(m) or not 0 <= m <= 10:
            raise ValueError("m must be finite and in [0, 10]")
        self.n, self.m = n, float(m)

    @staticmethod
    def _randint(high: int, generator: torch.Generator, device: torch.device) -> int:
        return int(torch.randint(high, (), generator=generator, device=device).item())

    @staticmethod
    def _uniform(generator: torch.Generator, device: torch.device) -> float:
        return float(torch.rand((), generator=generator, device=device).item())

    @staticmethod
    def _validate(x: torch.Tensor, generator: torch.Generator) -> None:
        if not isinstance(x, torch.Tensor) or x.ndim != 4 or x.shape[1] not in (1, 3):
            raise ValueError("input must be NCHW with 1 or 3 channels")
        if not x.is_floating_point() or not bool(torch.isfinite(x).all()) or bool(((x < 0) | (x > 1)).any()):
            raise ValueError("input must be finite floating point in [0, 1]")
        if not isinstance(generator, torch.Generator):
            raise ValueError("generator must be a torch.Generator")

    def _op(self, x: torch.Tensor, name: str, level: float, generator: torch.Generator) -> torch.Tensor:
        device = x.device
        sign = -1.0 if self._randint(2, generator, device) == 0 else 1.0
        level = max(0., min(30., float(level)))
        if name == "identity": return x
        if name == "brightness": return (x * (1 + sign * .9 * level / 10)).clamp(0, 1)
        if name == "contrast":
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            return (mean + (x - mean) * (1 + sign * .9 * level / 10)).clamp(0, 1)
        if name == "color":
            if x.shape[1] == 1: return x
            gray = (x[:, :1] * .2989 + x[:, 1:2] * .587 + x[:, 2:3] * .114)
            return (gray + (x - gray) * (1 + sign * .9 * level / 10)).clamp(0, 1)
        if name == "sharpness":
            blur = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
            return (blur + (x - blur) * (1 + sign * .9 * level / 10)).clamp(0, 1)
        if name == "autocontrast":
            lo, hi = x.amin((2, 3), keepdim=True), x.amax((2, 3), keepdim=True)
            return torch.where(hi > lo, (x-lo)/(hi-lo).clamp_min(1e-12), x).clamp(0, 1)
        if name == "equalize":

            flat = x.flatten(2)
            ranks = flat.argsort(dim=-1).argsort(dim=-1).to(x.dtype)
            return (ranks / max(1, flat.shape[-1]-1)).view_as(x)
        if name == "solarize": return torch.where(x >= 1 - level/10, 1-x, x)
        if name == "posterize":
            bits = max(1, 8 - round(4 * level / 10))
            step = 1 / (2 ** bits - 1)
            return (torch.round(x / step) * step).clamp(0, 1)
        if name in ("rotate", "shear-x", "shear-y", "translate-x", "translate-y"):
            strength = level / 10
            theta = torch.eye(2, 3, device=device, dtype=x.dtype).unsqueeze(0).repeat(x.shape[0], 1, 1)
            if name == "rotate":
                angle = sign * 30 * strength * math.pi / 180
                theta[:, 0, 0] = math.cos(angle); theta[:, 0, 1] = -math.sin(angle)
                theta[:, 1, 0] = math.sin(angle); theta[:, 1, 1] = math.cos(angle)
            elif name == "shear-x": theta[:, 0, 1] = sign * .3 * strength
            elif name == "shear-y": theta[:, 1, 0] = sign * .3 * strength
            elif name == "translate-x": theta[:, 0, 2] = sign * .3 * strength
            else: theta[:, 1, 2] = sign * .3 * strength
            grid = F.affine_grid(theta, x.shape, align_corners=False)
            return F.grid_sample(x, grid, mode="nearest", padding_mode="zeros", align_corners=False).clamp(0, 1)
        raise ValueError(f"unknown operation: {name}")

    def forward(self, x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        if generator is None:
            generator = torch.default_generator
        self._validate(x, generator)
        for _ in range(self.n):
            op = self.OPS[self._randint(len(self.OPS), generator, x.device)]
            x = self._op(x, op, self.m, generator)
        return x
