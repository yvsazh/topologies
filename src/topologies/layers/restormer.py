import math

import torch
from torch import nn
from torch.nn import functional as F


def _positive_finite_factor(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        factor = float(value)
    except (OverflowError, ValueError):
        return None
    return factor if math.isfinite(factor) and factor > 0 else None


class Restormer(nn.Module):
    component_id = "restormer"
    PRESETS = {
        "tiny": {"dim": 4, "num_heads": 1, "expansion_factor": 2.66},
        "small": {"dim": 16, "num_heads": 2, "expansion_factor": 2.66},
        "base": {"dim": 48, "num_heads": 1, "expansion_factor": 2.66},
    }

    def __init__(self, dim=48, num_heads=1, expansion_factor=2.66):
        super().__init__()
        if type(dim) is not int or dim <= 0:
            raise ValueError("dim must be a positive integer")
        if type(num_heads) is not int or num_heads <= 0:
            raise ValueError("num_heads must be a positive integer")
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        expansion_factor = _positive_finite_factor(expansion_factor)
        if expansion_factor is None:
            raise ValueError("expansion_factor must be a finite positive number")
        hidden_dim = math.floor(dim * expansion_factor)
        if hidden_dim <= 0:
            raise ValueError("floor(dim * expansion_factor) must be positive")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.expansion_factor = expansion_factor
        self.hidden_dim = hidden_dim
        self.eps = 1e-5
        self.mdta_norm = nn.LayerNorm(dim, eps=self.eps, elementwise_affine=True)
        self.to_q = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_k = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_v = nn.Conv2d(dim, dim, 1, bias=False)
        self.q_depthwise = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.k_depthwise = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.v_depthwise = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.alpha = nn.Parameter(torch.ones(1))
        self.mdta_project = nn.Conv2d(dim, dim, 1, bias=False)
        self.gdfn_norm = nn.LayerNorm(dim, eps=self.eps, elementwise_affine=True)
        self.gdfn_expand_1 = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.gdfn_depthwise_1 = nn.Conv2d(
            hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False
        )
        self.gdfn_expand_2 = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.gdfn_depthwise_2 = nn.Conv2d(
            hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False
        )
        self.gdfn_project = nn.Conv2d(hidden_dim, dim, 1, bias=False)

    def _normalize(self, x, norm):
        return norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def _mdta(self, x):
        batch, _, height, width = x.shape
        spatial = height * width
        normalized = self._normalize(x, self.mdta_norm)
        query = self.q_depthwise(self.to_q(normalized))
        key = self.k_depthwise(self.to_k(normalized))
        value = self.v_depthwise(self.to_v(normalized))
        query = query.reshape(batch, self.num_heads, self.head_dim, spatial).transpose(-2, -1)
        key = key.reshape(batch, self.num_heads, self.head_dim, spatial)
        value = value.reshape(batch, self.num_heads, self.head_dim, spatial).transpose(-2, -1)
        attention = torch.softmax(torch.matmul(key, query) / self.alpha, dim=-1)
        attended = torch.matmul(value, attention)
        attended = attended.transpose(-2, -1).reshape(batch, self.dim, height, width)
        return x + self.mdta_project(attended), attention

    def _gdfn(self, x):
        normalized = self._normalize(x, self.gdfn_norm)
        branch_1 = self.gdfn_depthwise_1(self.gdfn_expand_1(normalized))
        branch_2 = self.gdfn_depthwise_2(self.gdfn_expand_2(normalized))
        gated = F.gelu(branch_1) * branch_2
        return x + self.gdfn_project(gated), gated

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("input must be a torch.Tensor")
        if x.ndim != 4:
            raise ValueError("input must have shape [batch, channels, height, width]")
        if x.shape[1] != self.dim:
            raise ValueError(f"input channel count must be {self.dim}")
        if any(size <= 0 for size in (x.shape[0], x.shape[2], x.shape[3])):
            raise ValueError("input batch and spatial dimensions must be positive")
        if not torch.is_floating_point(x):
            raise ValueError("input must have a floating-point dtype")
        x, _ = self._mdta(x)
        x, _ = self._gdfn(x)
        return x
