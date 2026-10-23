import math

import torch
from torch import nn
from torch.nn import functional as F


class SNGP(nn.Module):
    component_id = "sngp"

    def __init__(
        self,
        in_features,
        num_classes,
        num_residual_blocks=11,
        num_random_features=1024,
        spectral_bound=0.95,
        power_iterations=1,
        ridge_factor=0.001,
        predictive_samples=10,
        activation="relu",
    ):
        super().__init__()
        dimensions = {
            "in_features": in_features,
            "num_classes": num_classes,
            "num_random_features": num_random_features,
            "power_iterations": power_iterations,
            "predictive_samples": predictive_samples,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(num_residual_blocks, bool) or not isinstance(num_residual_blocks, int) or num_residual_blocks < 0:
            raise ValueError("num_residual_blocks must be a nonnegative integer")
        if (
            isinstance(spectral_bound, bool)
            or not isinstance(spectral_bound, (int, float))
            or not math.isfinite(spectral_bound)
            or spectral_bound <= 0
        ):
            raise ValueError("spectral_bound must be a finite positive number")
        if (
            isinstance(ridge_factor, bool)
            or not isinstance(ridge_factor, (int, float))
            or not math.isfinite(ridge_factor)
            or ridge_factor <= 0
        ):
            raise ValueError("ridge_factor must be a finite positive number")
        if not isinstance(activation, str) or activation not in {"relu", "tanh"}:
            raise ValueError("activation must be 'relu' or 'tanh'")
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_residual_blocks = num_residual_blocks
        self.num_random_features = num_random_features
        self.spectral_bound = float(spectral_bound)
        self.power_iterations = power_iterations
        self.ridge_factor = float(ridge_factor)
        self.predictive_samples = predictive_samples
        self.activation = activation
        self.residual_layers = nn.ModuleList(
            nn.Linear(in_features, in_features) for _ in range(num_residual_blocks)
        )
        initial_vector = torch.full(
            (num_residual_blocks, in_features), 1.0 / math.sqrt(in_features)
        )
        self.register_buffer("spectral_vectors", initial_vector)
        self.register_buffer("rff_weight", torch.randn(num_random_features, in_features))
        self.register_buffer("rff_bias", torch.rand(num_random_features) * (2 * math.pi))
        self.output_weights = nn.Parameter(torch.empty(num_classes, num_random_features))
        nn.init.normal_(self.output_weights, std=1.0 / math.sqrt(num_random_features))
        self.register_buffer(
            "precision_matrix",
            torch.eye(num_random_features).expand(num_classes, -1, -1).clone() * self.ridge_factor,
        )

    def _validate_inputs(self, inputs, width, name):
        if not isinstance(inputs, torch.Tensor) or inputs.ndim < 1 or inputs.shape[-1] != width:
            raise ValueError(f"{name} must be a torch.Tensor with final dimension {width}")

    def hidden_representation(self, inputs):
        self._validate_inputs(inputs, self.in_features, "inputs")
        hidden = inputs
        for layer in self.residual_layers:
            residual = layer(hidden)
            if self.activation == "relu":
                residual = F.relu(residual)
            else:
                residual = torch.tanh(residual)
            hidden = hidden + residual
        return hidden

    def random_fourier_features(self, hidden):
        self._validate_inputs(hidden, self.in_features, "hidden")
        projection = -hidden @ self.rff_weight.transpose(-2, -1) + self.rff_bias
        return math.sqrt(2.0 / self.num_random_features) * torch.cos(projection)

    def logits_from_features(self, features):
        self._validate_inputs(features, self.num_random_features, "features")
        return features @ self.output_weights.transpose(0, 1)

    def forward(self, inputs):
        hidden = self.hidden_representation(inputs)
        return self.logits_from_features(self.random_fourier_features(hidden))

    def negative_log_posterior(self, logits, targets):
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
            raise ValueError("logits must have shape [batch, num_classes]")
        if logits.shape[1] != self.num_classes:
            raise ValueError("logits class dimension must equal num_classes")
        if not isinstance(targets, torch.Tensor) or targets.ndim != 1:
            raise ValueError("targets must have shape [batch]")
        if targets.shape[0] != logits.shape[0]:
            raise ValueError("targets and logits must have the same batch size")
        return F.cross_entropy(logits, targets.long()) + 0.5 * self.output_weights.square().sum()

    def compute_precision(self, features, probabilities):
        self._validate_inputs(features, self.num_random_features, "features")
        if features.ndim != 2:
            raise ValueError("features must have shape [batch, num_random_features]")
        if not isinstance(probabilities, torch.Tensor) or probabilities.ndim != 2:
            raise ValueError("probabilities must have shape [batch, num_classes]")
        if probabilities.shape != (features.shape[0], self.num_classes):
            raise ValueError("probabilities must match feature batch size and num_classes")
        weights = probabilities * (1 - probabilities)
        precision = torch.eye(
            self.num_random_features, device=features.device, dtype=features.dtype
        ).expand(self.num_classes, -1, -1) * self.ridge_factor
        return precision + torch.einsum("bk,bd,be->kde", weights, features, features)

    def reset_precision(self):
        with torch.no_grad():
            identity = torch.eye(
                self.num_random_features,
                device=self.precision_matrix.device,
                dtype=self.precision_matrix.dtype,
            )
            self.precision_matrix.copy_(identity.unsqueeze(0).expand(self.num_classes, -1, -1))
            self.precision_matrix.mul_(self.ridge_factor)

    @torch.no_grad()
    def accumulate_precision(self, inputs):
        self._validate_inputs(inputs, self.in_features, "inputs")
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)
        features = self.random_fourier_features(self.hidden_representation(inputs))
        probabilities = self.logits_from_features(features).softmax(dim=-1)
        contribution = torch.einsum(
            "bk,bd,be->kde", probabilities * (1 - probabilities), features, features
        )
        self.precision_matrix.add_(contribution)

    @property
    def posterior_covariance(self):
        return torch.linalg.inv(self.precision_matrix)

    def predictive_variance(self, features, covariance):
        self._validate_inputs(features, self.num_random_features, "features")
        if features.ndim != 2:
            raise ValueError("features must have shape [batch, num_random_features]")
        expected = (self.num_classes, self.num_random_features, self.num_random_features)
        if not isinstance(covariance, torch.Tensor) or tuple(covariance.shape) != expected:
            raise ValueError(f"covariance must have shape {expected}")
        variance = torch.einsum("bd,kde,be->bk", features, covariance, features)
        return variance.clamp_min(0)

    def posterior_moments(self, inputs):
        self._validate_inputs(inputs, self.in_features, "inputs")
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)
        features = self.random_fourier_features(self.hidden_representation(inputs))
        logits = self.logits_from_features(features)
        variance = self.predictive_variance(features, self.posterior_covariance)
        return logits, variance

    def predictive_probabilities(self, inputs, samples=None):
        sample_count = self.predictive_samples if samples is None else samples
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
            raise ValueError("samples must be a positive integer")
        mean, variance = self.posterior_moments(inputs)
        noise = torch.randn(
            (sample_count, *mean.shape), device=mean.device, dtype=mean.dtype
        )
        logits = mean.unsqueeze(0) + noise * variance.clamp_min(0).sqrt().unsqueeze(0)
        return logits.softmax(dim=-1).mean(dim=0)

    @torch.no_grad()
    def project_spectral_weight(self, weight, initial_vector=None):
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise ValueError("weight must be a rank-two tensor")
        if weight.shape[0] != weight.shape[1]:
            raise ValueError("spectral projection requires a square weight matrix")
        if initial_vector is None:
            vector = torch.ones(weight.shape[1], device=weight.device, dtype=weight.dtype)
        else:
            if not isinstance(initial_vector, torch.Tensor) or initial_vector.shape != (weight.shape[1],):
                raise ValueError("initial_vector must match the weight's input dimension")
            vector = initial_vector.to(device=weight.device, dtype=weight.dtype)
        norm = torch.linalg.vector_norm(vector)
        if not torch.isfinite(norm) or norm == 0:
            raise ValueError("initial_vector must have a finite nonzero norm")
        vector = vector / norm
        left = torch.zeros_like(vector)
        for _ in range(self.power_iterations):
            left = weight @ vector
            left_norm = torch.linalg.vector_norm(left)
            if left_norm == 0:
                return 0.0
            left = left / left_norm
            vector = weight.transpose(0, 1) @ left
            vector_norm = torch.linalg.vector_norm(vector)
            if vector_norm == 0:
                return 0.0
            vector = vector / vector_norm
        estimate = torch.abs(left @ (weight @ vector))
        if estimate > self.spectral_bound:
            weight.mul_(self.spectral_bound / estimate)
        if initial_vector is not None:
            initial_vector.copy_(vector.to(device=initial_vector.device, dtype=initial_vector.dtype))
        return estimate.item()

    @torch.no_grad()
    def project_spectral_weights(self):
        for index, layer in enumerate(self.residual_layers):
            vector = self.spectral_vectors[index]
            self.project_spectral_weight(layer.weight, vector)
            vector_norm = torch.linalg.vector_norm(vector)
            if vector_norm > 0:
                vector.div_(vector_norm)
