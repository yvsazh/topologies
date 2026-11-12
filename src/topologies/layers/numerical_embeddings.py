import math

import torch
from torch import nn


class NumericalEmbeddings(nn.Module):
    component_id = "numerical-embeddings"

    def __init__(
        self,
        n_features: int,
        mode: str = "periodic",
        n_frequencies: int | None = None,
        sigma: float | None = None,
        bin_edges=None,
    ) -> None:
        super().__init__()
        if isinstance(n_features, bool) or not isinstance(n_features, int) or n_features < 1:
            raise ValueError("n_features must be a positive integer")
        if not isinstance(mode, str) or mode not in {"periodic", "ple"}:
            raise ValueError("mode must be 'periodic' or 'ple'")
        self.n_features = n_features
        self.mode = mode
        if mode == "periodic":
            if isinstance(n_frequencies, bool) or not isinstance(n_frequencies, int) or n_frequencies < 1:
                raise ValueError("n_frequencies must be a positive integer for periodic mode")
            if isinstance(sigma, bool) or not isinstance(sigma, (int, float)):
                raise ValueError("sigma must be a finite positive number for periodic mode")
            if not math.isfinite(sigma) or sigma <= 0:
                raise ValueError("sigma must be a finite positive number for periodic mode")
            if bin_edges is not None:
                raise ValueError("bin_edges are only valid in ple mode")
            self.n_frequencies = n_frequencies
            self.sigma = float(sigma)
            self.coefficients = nn.Parameter(torch.randn(n_features, n_frequencies) * self.sigma)
            self.register_buffer("bin_edges", None)
            self.n_bins = None
        else:
            if n_frequencies is not None or sigma is not None:
                raise ValueError("n_frequencies and sigma are only valid in periodic mode")
            if bin_edges is None:
                raise ValueError("bin_edges are required for ple mode")
            try:
                edges = torch.as_tensor(bin_edges, dtype=torch.get_default_dtype())
            except (TypeError, ValueError, RuntimeError) as error:
                raise ValueError("bin_edges must be a finite rank-2 numeric array") from error
            if edges.ndim != 2 or edges.shape[0] != n_features or edges.shape[1] < 2:
                raise ValueError("bin_edges must have shape [n_features, T+1] with at least two edges")
            if not torch.isfinite(edges).all():
                raise ValueError("bin_edges must contain only finite values")
            if not torch.all(edges[:, 1:] > edges[:, :-1]):
                raise ValueError("bin_edges must be strictly increasing for each feature")
            self.n_frequencies = None
            self.sigma = None
            self.register_parameter("coefficients", None)
            self.register_buffer("bin_edges", edges)
            self.n_bins = edges.shape[1] - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"x must be a rank-2 tensor with shape [B, {self.n_features}]")
        if self.mode == "periodic":
            dtype = x.dtype if x.is_floating_point() else self.coefficients.dtype
            coefficients = self.coefficients.to(dtype=dtype)
            phase = (2.0 * torch.pi) * x.unsqueeze(-1) * coefficients
            return torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
        dtype = x.dtype if x.is_floating_point() else self.bin_edges.dtype
        edges = self.bin_edges.to(dtype=dtype)
        lower = edges[:, :-1]
        upper = edges[:, 1:]
        values = x.unsqueeze(-1)
        encoded = (values - lower) / (upper - lower)
        bins = torch.arange(self.n_bins, device=x.device)
        below = (values < lower) & (bins > 0)
        above = (values >= upper) & (bins < self.n_bins - 1)
        return torch.where(below, torch.zeros_like(encoded), torch.where(above, torch.ones_like(encoded), encoded))
