import math

import torch
from torch import nn
from torch.nn import functional as F


class NeuralProcess(nn.Module):
    component_id = "neural-process"
    PRESETS = {
        "tiny": {"hidden_dim": 16, "latent_dim": 8, "hidden_layers": 2},
        "small": {"hidden_dim": 64, "latent_dim": 32, "hidden_layers": 2},
        "base": {"hidden_dim": 128, "latent_dim": 64, "hidden_layers": 2},
    }

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=128,
        latent_dim=64,
        hidden_layers=2,
        activation="relu",
        observation_noise=None,
        latent_scale_floor=1e-4,
        latent_samples=1,
    ):
        super().__init__()
        for name, value in (
            ("input_dim", input_dim),
            ("output_dim", output_dim),
            ("hidden_dim", hidden_dim),
            ("latent_dim", latent_dim),
            ("hidden_layers", hidden_layers),
            ("latent_samples", latent_samples),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(activation, str) or activation.lower() not in {"relu", "tanh", "gelu"}:
            raise ValueError("activation must be one of 'relu', 'tanh', or 'gelu'")
        if (
            isinstance(observation_noise, bool)
            or not isinstance(observation_noise, (int, float))
            or not math.isfinite(float(observation_noise))
            or observation_noise <= 0
        ):
            raise ValueError("observation_noise must be a finite positive scalar")
        if (
            isinstance(latent_scale_floor, bool)
            or not isinstance(latent_scale_floor, (int, float))
            or not math.isfinite(float(latent_scale_floor))
            or latent_scale_floor <= 0
        ):
            raise ValueError("latent_scale_floor must be a finite positive scalar")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        self.activation = activation.lower()
        self.observation_noise = float(observation_noise)
        self.latent_scale_floor = float(latent_scale_floor)
        self.latent_samples = latent_samples
        self.encoder = self._make_mlp(input_dim + output_dim, hidden_dim)
        self.latent_mean = nn.Linear(hidden_dim, latent_dim)
        self.latent_scale = nn.Linear(hidden_dim, latent_dim)
        self.decoder = self._make_mlp(input_dim + latent_dim, output_dim)

    def _make_mlp(self, input_size, output_size):
        activation_type = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[self.activation]
        layers = []
        width = input_size
        for _ in range(self.hidden_layers):
            layers.extend((nn.Linear(width, self.hidden_dim), activation_type()))
            width = self.hidden_dim
        layers.append(nn.Linear(width, output_size))
        return nn.Sequential(*layers)

    @staticmethod
    def aggregate_context(encoded_context):
        if not isinstance(encoded_context, torch.Tensor) or encoded_context.ndim < 2:
            raise ValueError("encoded_context must include a nonempty context axis")
        if encoded_context.shape[-2] == 0:
            raise ValueError("encoded_context must not be empty")
        return encoded_context.mean(dim=-2)

    def _validate_pair(self, x, y, name):
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError(f"{name}_x and {name}_y must be tensors")
        if x.ndim != 3 or y.ndim != 3:
            raise ValueError(f"{name}_x and {name}_y must have shape [batch, points, features]")
        if not x.is_floating_point() or not y.is_floating_point():
            raise ValueError(f"{name}_x and {name}_y must have floating-point dtype")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"{name}_x and {name}_y batch dimensions must match")
        if x.shape[1] != y.shape[1]:
            raise ValueError(f"{name}_x and {name}_y point dimensions must match")
        if x.shape[1] == 0:
            raise ValueError(f"{name} set must not be empty")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"{name}_x last dimension must equal input_dim={self.input_dim}")
        if y.shape[-1] != self.output_dim:
            raise ValueError(f"{name}_y last dimension must equal output_dim={self.output_dim}")

    def encode_context(self, context_x, context_y):
        self._validate_pair(context_x, context_y, "context")
        encoded = self.encoder(torch.cat((context_x, context_y), dim=-1))
        representation = self.aggregate_context(encoded)
        mean = self.latent_mean(representation)
        scale = F.softplus(self.latent_scale(representation)) + self.latent_scale_floor
        return representation, mean, scale

    @staticmethod
    def diagonal_gaussian_kl(q_mean, q_std, p_mean, p_std):
        if not all(isinstance(value, torch.Tensor) for value in (q_mean, q_std, p_mean, p_std)):
            raise ValueError("Gaussian means and standard deviations must be tensors")
        if not (q_mean.shape == q_std.shape == p_mean.shape == p_std.shape):
            raise ValueError("Gaussian means and standard deviations must have matching shapes")
        if bool((q_std <= 0).any()) or bool((p_std <= 0).any()):
            raise ValueError("Gaussian standard deviations must be positive")
        q_variance = q_std.square()
        p_variance = p_std.square()
        terms = torch.log(p_variance / q_variance)
        terms = terms + (q_variance + (q_mean - p_mean).square()) / p_variance - 1.0
        return 0.5 * terms.sum(dim=-1)

    @staticmethod
    def gaussian_log_likelihood(target, prediction_mean, observation_noise):
        if (
            isinstance(observation_noise, bool)
            or not isinstance(observation_noise, (int, float))
            or not math.isfinite(float(observation_noise))
            or observation_noise <= 0
        ):
            raise ValueError("observation_noise must be a finite positive scalar")
        try:
            target, prediction_mean = torch.broadcast_tensors(target, prediction_mean)
        except RuntimeError as error:
            raise ValueError("target and prediction_mean must have broadcastable shapes") from error
        variance = float(observation_noise) ** 2
        constant = math.log(2.0 * math.pi * variance)
        return (-0.5 * (constant + (target - prediction_mean).square() / variance)).sum(dim=-1)

    @classmethod
    def conditional_negative_elbo(cls, log_likelihood, q_mean, q_std, p_mean, p_std):
        if not isinstance(log_likelihood, torch.Tensor):
            raise TypeError("log_likelihood must be a tensor")
        total_log_likelihood = (
            log_likelihood if log_likelihood.ndim == 0 else log_likelihood.sum(dim=-1)
        )
        divergence = cls.diagonal_gaussian_kl(q_mean, q_std, p_mean, p_std)
        return -(total_log_likelihood - divergence).mean()

    def _decode(self, target_x, latent):
        repeated_latent = latent.unsqueeze(1).expand(-1, target_x.shape[1], -1)
        return self.decoder(torch.cat((target_x, repeated_latent), dim=-1))

    def forward(self, context_x, context_y, target_x, target_y=None, *, sample=True, return_stats=False):
        self._validate_pair(context_x, context_y, "context")
        if not isinstance(target_x, torch.Tensor) or target_x.ndim != 3:
            raise ValueError("target_x must have shape [batch, points, input_dim]")
        if not target_x.is_floating_point():
            raise ValueError("target_x must have floating-point dtype")
        if target_x.shape[0] != context_x.shape[0]:
            raise ValueError("context and target batch dimensions must match")
        if target_x.shape[1] == 0:
            raise ValueError("target set must not be empty")
        if target_x.shape[-1] != self.input_dim:
            raise ValueError(f"target_x last dimension must equal input_dim={self.input_dim}")
        if not isinstance(sample, bool):
            raise TypeError("sample must be a boolean")
        if not isinstance(return_stats, bool):
            raise TypeError("return_stats must be a boolean")
        if target_y is not None:
            self._validate_pair(target_x, target_y, "target")
        context_representation, p_mean, p_std = self.encode_context(context_x, context_y)
        if target_y is None:
            q_mean, q_std = p_mean, p_std
        else:
            combined_x = torch.cat((context_x, target_x), dim=1)
            combined_y = torch.cat((context_y, target_y), dim=1)
            _, q_mean, q_std = self.encode_context(combined_x, combined_y)
        sample_count = self.latent_samples if sample else 1
        predictions = []
        for _ in range(sample_count):
            latent = q_mean + q_std * torch.randn_like(q_std) if sample else q_mean
            predictions.append(self._decode(target_x, latent))
        prediction_samples = torch.stack(predictions, dim=0)
        prediction_mean = prediction_samples.mean(dim=0)
        result = {
            "mean": prediction_mean,
            "observation_std": prediction_mean.new_full(prediction_mean.shape, self.observation_noise),
            "context_representation": context_representation,
            "context_mean": p_mean,
            "context_std": p_std,
            "latent_mean": q_mean,
            "latent_std": q_std,
            "prediction_samples": prediction_samples,
        }
        if target_y is not None:
            sampled_log_likelihood = self.gaussian_log_likelihood(
                target_y.unsqueeze(0), prediction_samples, self.observation_noise
            )
            log_likelihood = sampled_log_likelihood.mean(dim=0)
            divergence = self.diagonal_gaussian_kl(q_mean, q_std, p_mean, p_std)
            per_example_loss = -(log_likelihood.sum(dim=-1) - divergence)
            loss = self.conditional_negative_elbo(log_likelihood, q_mean, q_std, p_mean, p_std)
            result.update({
                "log_likelihood": log_likelihood,
                "kl": divergence,
                "per_example_loss": per_example_loss,
                "loss": loss,
                "objective": loss,
            })
        if return_stats:
            return result
        if target_y is not None:
            return result["per_example_loss"].reshape(-1, 1)
        return prediction_mean
