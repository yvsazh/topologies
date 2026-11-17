import math

import torch
from torch import nn


class SparseGPVariationalBound(nn.Module):
    component_id = "sparse-gp"

    def __init__(
        self,
        input_dim,
        num_inducing=8,
        lengthscale=1.0,
        outputscale=1.0,
        noise_variance=0.1,
        jitter=1e-6,
    ):
        super().__init__()
        if not isinstance(input_dim, int) or isinstance(input_dim, bool) or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if not isinstance(num_inducing, int) or isinstance(num_inducing, bool) or num_inducing <= 0:
            raise ValueError("num_inducing must be a positive integer")
        for name, value in (
            ("lengthscale", lengthscale),
            ("outputscale", outputscale),
            ("noise_variance", noise_variance),
        ):
            valid_type = isinstance(value, (int, float)) and not isinstance(value, bool)
            try:
                valid_value = math.isfinite(float(value)) and float(value) > 0
            except (TypeError, ValueError, OverflowError):
                valid_value = False
            if not valid_type or not valid_value:
                raise ValueError(f"{name} must be a finite positive number")
        valid_jitter_type = isinstance(jitter, (int, float)) and not isinstance(jitter, bool)
        try:
            valid_jitter = math.isfinite(float(jitter)) and float(jitter) >= 0
        except (TypeError, ValueError, OverflowError):
            valid_jitter = False
        if not valid_jitter_type or not valid_jitter:
            raise ValueError("jitter must be a finite nonnegative number")

        generator = torch.Generator(device="cpu").manual_seed(0)
        self.inducing_points = nn.Parameter(
            torch.randn(num_inducing, input_dim, generator=generator)
        )
        self.log_lengthscale = nn.Parameter(torch.tensor(math.log(float(lengthscale))))
        self.log_outputscale = nn.Parameter(torch.tensor(math.log(float(outputscale))))
        self.log_noise_variance = nn.Parameter(torch.tensor(math.log(float(noise_variance))))
        self.input_dim = input_dim
        self.num_inducing = num_inducing
        self.jitter = float(jitter)

    @property
    def lengthscale(self):
        return self.log_lengthscale.exp()

    @property
    def outputscale(self):
        return self.log_outputscale.exp()

    @property
    def noise_variance(self):
        return self.log_noise_variance.exp()

    def _validate_scales(self):
        for name, value in (
            ("lengthscale", self.lengthscale),
            ("outputscale", self.outputscale),
            ("noise_variance", self.noise_variance),
        ):
            if not bool(torch.isfinite(value)) or not bool(value > 0):
                raise ValueError(f"{name} must remain finite and positive")

    def _validate_training_data(self, train_x, train_y):
        if not isinstance(train_x, torch.Tensor) or train_x.ndim != 2:
            raise ValueError("train_x must be a tensor with shape [N, input_dim]")
        if train_x.shape[0] == 0:
            raise ValueError("train_x and train_y must contain non-empty data")
        if train_x.shape[1] != self.input_dim:
            raise ValueError("train_x input_dim must match the configured input_dim")
        if not train_x.is_floating_point():
            raise ValueError("train_x must have a floating-point dtype")
        if not isinstance(train_y, torch.Tensor):
            raise TypeError("train_y must be a floating-point tensor")
        if train_y.ndim == 2 and train_y.shape[1] == 1:
            train_y = train_y[:, 0]
        if train_y.ndim != 1 or train_y.shape[0] != train_x.shape[0]:
            raise ValueError("train_y must have shape [N] or [N, 1]")
        if not train_y.is_floating_point():
            raise ValueError("train_y must have a floating-point dtype")
        if train_x.dtype != self.inducing_points.dtype or train_x.device != self.inducing_points.device:
            raise ValueError("train_x must match the module dtype and device")
        if train_y.dtype != train_x.dtype or train_y.device != train_x.device:
            raise ValueError("train_y must match train_x dtype and device")
        if not bool(torch.isfinite(train_x).all()) or not bool(torch.isfinite(train_y).all()):
            raise ValueError("train_x and train_y must contain only finite values")
        self._validate_scales()
        return train_x, train_y

    def _kernel(self, left, right):
        squared_distance = (left.unsqueeze(1) - right.unsqueeze(0)).square().sum(dim=-1)
        return self.outputscale * torch.exp(-0.5 * squared_distance / self.lengthscale.square())

    def _covariances(self, train_x):
        identity = torch.eye(
            self.num_inducing,
            dtype=train_x.dtype,
            device=train_x.device,
        )
        k_mm = self._kernel(self.inducing_points, self.inducing_points)
        k_mm = k_mm + self.jitter * identity
        k_nm = self._kernel(train_x, self.inducing_points)
        chol_mm = torch.linalg.cholesky(k_mm)
        noise = self.noise_variance
        r = k_mm + (k_nm.transpose(0, 1) @ k_nm) / noise
        chol_r = torch.linalg.cholesky(r)
        return k_mm, k_nm, chol_mm, chol_r

    def forward(self, train_x, train_y):
        if isinstance(train_x, torch.Tensor) and train_x.ndim == 3:
            if train_x.shape[0] == 0:
                raise ValueError("batched train_x must contain at least one dataset")
            if isinstance(train_y, torch.Tensor) and train_y.ndim == 3 and train_y.shape[-1] == 1:
                train_y = train_y[..., 0]
            if (
                not isinstance(train_y, torch.Tensor)
                or train_y.ndim != 2
                or train_y.shape[:2] != train_x.shape[:2]
            ):
                raise ValueError("batched train_y must have shape [B, N] or [B, N, 1]")
            return torch.stack(
                [self._forward_single(train_x[index], train_y[index])
                 for index in range(train_x.shape[0])]
            )
        return self._forward_single(train_x, train_y)

    def _forward_single(self, train_x, train_y):
        train_x, train_y = self._validate_training_data(train_x, train_y)
        _k_mm, k_nm, chol_mm, chol_r = self._covariances(train_x)
        noise = self.noise_variance
        cross_y = k_nm.transpose(0, 1) @ train_y
        solved_y = torch.cholesky_solve(cross_y.unsqueeze(1), chol_r).squeeze(1)
        logdet_kmm = 2.0 * torch.log(torch.diagonal(chol_mm)).sum()
        logdet_r = 2.0 * torch.log(torch.diagonal(chol_r)).sum()
        logdet_c = train_x.shape[0] * noise.log() + logdet_r - logdet_kmm
        quadratic = train_y.square().sum() / noise - cross_y.dot(solved_y) / noise.square()
        trace_projection = torch.cholesky_solve(k_nm.transpose(0, 1) @ k_nm, chol_mm)
        trace_residual = train_x.shape[0] * self.outputscale - torch.diagonal(trace_projection).sum()
        return -0.5 * (
            train_x.shape[0] * math.log(2.0 * math.pi) + logdet_c + quadratic
        ) - 0.5 * trace_residual / noise

    def predict(self, query_x, train_x, train_y, observation_noise=False):
        train_x, train_y = self._validate_training_data(train_x, train_y)
        if not isinstance(query_x, torch.Tensor) or query_x.ndim != 2:
            raise ValueError("query_x must be a tensor with shape [Q, input_dim]")
        if query_x.shape[0] == 0:
            raise ValueError("query_x must contain at least one query")
        if query_x.shape[1] != self.input_dim:
            raise ValueError("query_x input_dim must match the configured input_dim")
        if not query_x.is_floating_point():
            raise ValueError("query_x must have a floating-point dtype")
        if query_x.dtype != train_x.dtype or query_x.device != train_x.device:
            raise ValueError("query_x must match train_x dtype and device")
        if not bool(torch.isfinite(query_x).all()):
            raise ValueError("query_x must contain only finite values")
        if not isinstance(observation_noise, bool):
            raise TypeError("observation_noise must be a boolean")

        k_mm, k_nm, chol_mm, chol_r = self._covariances(train_x)
        k_qm = self._kernel(query_x, self.inducing_points)
        cross_y = k_nm.transpose(0, 1) @ train_y
        variational_mean = k_mm @ torch.cholesky_solve(cross_y.unsqueeze(1), chol_r)
        variational_mean = variational_mean / self.noise_variance
        solved_mean = torch.cholesky_solve(variational_mean, chol_mm)
        mean = (k_qm @ solved_mean).squeeze(1)
        projected_prior = torch.cholesky_solve(k_qm.transpose(0, 1), chol_mm)
        posterior_covariance = torch.cholesky_solve(k_qm.transpose(0, 1), chol_r)
        variance = self.outputscale - (k_qm * projected_prior.transpose(0, 1)).sum(dim=1)
        variance = variance + (k_qm * posterior_covariance.transpose(0, 1)).sum(dim=1)
        if observation_noise:
            variance = variance + self.noise_variance
        return mean, variance
