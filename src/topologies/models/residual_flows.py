import math

import torch
from torch import nn
from torch.nn import functional as F


class ResidualFlows(nn.Module):
    architecture_id = "residual-flows"
    PRESETS = {
        "tiny": {
            "features": 2,
            "hidden_dim": 8,
            "num_blocks": 1,
            "spectral_bound": 0.98,
            "exact_terms": 2,
            "tail_stop_probability": 0.5,
            "lip_swish_beta": 1.0,
        },
        "small": {
            "features": 2,
            "hidden_dim": 64,
            "num_blocks": 2,
            "spectral_bound": 0.98,
            "exact_terms": 2,
            "tail_stop_probability": 0.5,
            "lip_swish_beta": 1.0,
        },
    }

    def __init__(
        self,
        features,
        hidden_dim=8,
        num_blocks=1,
        spectral_bound=0.98,
        exact_terms=2,
        tail_stop_probability=0.5,
        lip_swish_beta=1.0,
    ):
        super().__init__()
        if not isinstance(features, int) or isinstance(features, bool) or features <= 0:
            raise ValueError("features must be a positive integer")
        if not isinstance(hidden_dim, int) or isinstance(hidden_dim, bool) or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        if not isinstance(num_blocks, int) or isinstance(num_blocks, bool) or num_blocks <= 0:
            raise ValueError("num_blocks must be a positive integer")
        if (
            not isinstance(spectral_bound, (int, float))
            or isinstance(spectral_bound, bool)
            or not math.isfinite(spectral_bound)
            or not 0 < spectral_bound < 1
        ):
            raise ValueError("spectral_bound must be between 0 and 1")
        if not isinstance(exact_terms, int) or isinstance(exact_terms, bool) or exact_terms <= 0:
            raise ValueError("exact_terms must be a positive integer")
        if (
            not isinstance(tail_stop_probability, (int, float))
            or isinstance(tail_stop_probability, bool)
            or not math.isfinite(tail_stop_probability)
            or not 0 < tail_stop_probability < 1
        ):
            raise ValueError("tail_stop_probability must be between 0 and 1")
        if (
            not isinstance(lip_swish_beta, (int, float))
            or isinstance(lip_swish_beta, bool)
            or not math.isfinite(lip_swish_beta)
            or not lip_swish_beta > 0
        ):
            raise ValueError("lip_swish_beta must be positive")
        self.features = features
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.spectral_bound = float(spectral_bound)
        self.exact_terms = exact_terms
        self.tail_stop_probability = float(tail_stop_probability)
        self.lip_swish_beta = float(lip_swish_beta)
        self.layers = nn.ModuleList(
            nn.ModuleList((nn.Linear(features, hidden_dim), nn.Linear(hidden_dim, features)))
            for _ in range(num_blocks)
        )

    def _validate_input(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 2:
            raise ValueError("input must be a tensor with shape [batch, features]")
        if x.shape[0] <= 0 or x.shape[1] != self.features:
            raise ValueError(f"input shape must be [batch, {self.features}] with a positive batch")
        if not x.is_floating_point():
            raise ValueError("input must have a floating-point dtype")

    def _validate_probe(self, probe, x):
        if not isinstance(probe, torch.Tensor) or probe.shape != x.shape:
            raise ValueError("probe must have the same shape as the input")
        if probe.device != x.device or probe.dtype != x.dtype:
            raise ValueError("probe must match the input dtype and device")

    def _bounded_linear(self, layer, x):
        norm = torch.linalg.matrix_norm(layer.weight.detach(), ord=2)
        scale = (self.spectral_bound / norm.clamp_min(torch.finfo(layer.weight.dtype).tiny)).clamp(max=1.0)
        return F.linear(x, layer.weight * scale, layer.bias)

    def _residual(self, x, index):
        first, second = self.layers[index]
        hidden = self._bounded_linear(first, x)
        hidden = hidden * torch.sigmoid(self.lip_swish_beta * hidden) / 1.1
        return self._bounded_linear(second, hidden)

    def residual_block(self, x, index=0):
        self._validate_input(x)
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < self.num_blocks:
            raise ValueError("index must identify a residual block")
        return self._residual(x, index)

    def _series_logdet(self, residual, x, probe, num_terms, exact_terms, create_graph):
        vector = probe
        estimate = x.new_zeros(x.shape[0])
        for term in range(1, num_terms + 1):
            vector = torch.autograd.grad(
                residual,
                x,
                grad_outputs=vector,
                create_graph=create_graph,
                retain_graph=True,
            )[0]
            survival = 1.0
            if term > exact_terms:
                survival = (1.0 - self.tail_stop_probability) ** (term - exact_terms - 1)
            coefficient = (-1.0) ** (term + 1) / (term * survival)
            estimate = estimate + coefficient * (probe * vector).sum(dim=-1)
        return estimate

    def _validate_term_counts(self, num_terms, exact_terms):
        if not isinstance(num_terms, int) or isinstance(num_terms, bool) or num_terms <= 0:
            raise ValueError("num_terms must be a positive integer")
        if exact_terms is not None and (
            not isinstance(exact_terms, int)
            or isinstance(exact_terms, bool)
            or not 0 <= exact_terms <= num_terms
        ):
            raise ValueError("exact_terms must be an integer between zero and num_terms")

    def logdet_series(self, x, probe, num_terms, exact_terms=None):
        self._validate_input(x)
        self._validate_probe(probe, x)
        self._validate_term_counts(num_terms, exact_terms)
        create_graph = torch.is_grad_enabled()
        exact = num_terms if exact_terms is None else exact_terms
        with torch.enable_grad():
            current = x if x.requires_grad else x.detach().requires_grad_(True)
            estimate = current.new_zeros(current.shape[0])
            for index in range(self.num_blocks):
                residual = self._residual(current, index)
                estimate = estimate + self._series_logdet(
                    residual, current, probe, num_terms, exact, create_graph
                )
                current = current + residual
        return estimate if create_graph else estimate.detach()

    def _sample_tail_terms(self, generator, device):
        sample = torch.rand((), generator=generator, device=device).item()
        return math.floor(math.log1p(-sample) / math.log1p(-self.tail_stop_probability)) + 1

    def transform_with_logdet(self, x, generator=None, probe=None, tail_terms=None):
        self._validate_input(x)
        if probe is not None:
            self._validate_probe(probe, x)
        if tail_terms is not None and (
            not isinstance(tail_terms, int) or isinstance(tail_terms, bool) or tail_terms < 0
        ):
            raise ValueError("tail_terms must be a nonnegative integer")
        create_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            current = x if x.requires_grad else x.detach().requires_grad_(True)
            total_logdet = current.new_zeros(current.shape[0])
            for index in range(self.num_blocks):
                residual = self._residual(current, index)
                block_probe = probe
                if block_probe is None:
                    block_probe = torch.randn(
                        current.shape,
                        dtype=current.dtype,
                        device=current.device,
                        generator=generator,
                    )
                count = tail_terms
                if count is None:
                    count = self._sample_tail_terms(generator, current.device)
                num_terms = self.exact_terms + count
                total_logdet = total_logdet + self._series_logdet(
                    residual,
                    current,
                    block_probe,
                    num_terms,
                    self.exact_terms,
                    create_graph,
                )
                current = current + residual
        if create_graph:
            return current, total_logdet
        return current.detach(), total_logdet.detach()

    def transform(self, x, generator=None, probe=None, tail_terms=None):
        return self.transform_with_logdet(
            x, generator=generator, probe=probe, tail_terms=tail_terms
        )[0]

    def log_prob(self, x, generator=None, probe=None, tail_terms=None):
        transformed, logdet = self.transform_with_logdet(
            x, generator=generator, probe=probe, tail_terms=tail_terms
        )
        base_log_prob = -0.5 * (
            transformed.square().sum(dim=-1) + self.features * math.log(2 * math.pi)
        )
        return base_log_prob + logdet

    def forward(self, x, generator=None, probe=None, tail_terms=None):
        return self.log_prob(x, generator=generator, probe=probe, tail_terms=tail_terms)
