import math

import torch
from torch import nn
from torch.nn import functional as F


class DenoisingNeuralProcess(nn.Module):
    architecture_id = "denoising-neural-process"
    PRESETS = {
        "tiny": {
            "h_dim": 16,
            "r_dim": 8,
            "z_dim": 4,
            "x_noise_std": 0.1,
            "beta_kl": 1.0,
            "beta_denoise": 0.1,
        },
        "small": {
            "h_dim": 64,
            "r_dim": 64,
            "z_dim": 32,
            "x_noise_std": 0.1,
            "beta_kl": 1.0,
            "beta_denoise": 0.1,
        },
        "base": {
            "h_dim": 128,
            "r_dim": 128,
            "z_dim": 64,
            "x_noise_std": 0.1,
            "beta_kl": 1.0,
            "beta_denoise": 0.1,
        },
    }

    def __init__(
        self,
        x_dim,
        y_dim,
        h_dim=128,
        r_dim=128,
        z_dim=64,
        x_noise_std=0.1,
        beta_kl=1.0,
        beta_denoise=0.1,
        x_bounds=(-1.0, 1.0),
    ):
        super().__init__()
        for name, value in (("x_dim", x_dim), ("y_dim", y_dim), ("h_dim", h_dim),
                            ("r_dim", r_dim), ("z_dim", z_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (("beta_kl", beta_kl), ("beta_denoise", beta_denoise)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be a finite nonnegative number")
        if isinstance(x_noise_std, torch.Tensor):
            if x_noise_std.ndim > 1:
                raise ValueError("x_noise_std must be a positive scalar or length-x_dim vector")
            noise_values = x_noise_std.detach().cpu().reshape(-1).tolist()
        elif isinstance(x_noise_std, (tuple, list)):
            noise_values = list(x_noise_std)
        else:
            noise_values = [x_noise_std]
        if len(noise_values) not in (1, x_dim):
            raise ValueError("x_noise_std must be a positive scalar or length-x_dim vector")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value <= 0 for value in noise_values):
            raise ValueError("x_noise_std must be positive and finite")
        if isinstance(x_bounds, torch.Tensor):
            if x_bounds.ndim != 1:
                raise ValueError("x_bounds must contain two finite increasing values")
            bounds = x_bounds.detach().cpu().tolist()
        elif isinstance(x_bounds, (tuple, list)):
            bounds = list(x_bounds)
        else:
            bounds = []
        if (len(bounds) != 2 or any(isinstance(value, bool)
                                    or not isinstance(value, (int, float))
                                    or not math.isfinite(value) for value in bounds)
                or bounds[0] >= bounds[1]):
            raise ValueError("x_bounds must contain two finite increasing values")
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.beta_kl = float(beta_kl)
        self.beta_denoise = float(beta_denoise)
        self.x_bounds = (float(bounds[0]), float(bounds[1]))
        if len(noise_values) == 1:
            noise_values = noise_values * x_dim
        self.register_buffer("x_noise_std", torch.tensor(noise_values, dtype=torch.float32))
        self.denoiser = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )
        self.denoiser_loc = nn.Linear(h_dim, x_dim)
        self.denoiser_log_scale = nn.Linear(h_dim, x_dim)
        nn.init.normal_(self.denoiser_loc.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.denoiser_loc.bias)
        self.encoder = nn.Sequential(
            nn.Linear(x_dim + y_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, r_dim),
        )
        self.latent_hidden = nn.Linear(r_dim, r_dim)
        self.latent_mean = nn.Linear(r_dim, z_dim)
        self.latent_scale = nn.Linear(r_dim, z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(x_dim + z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )
        self.output_loc = nn.Linear(h_dim, y_dim)
        self.output_scale = nn.Linear(h_dim, y_dim)

    @staticmethod
    def diagonal_gaussian_kl(q_loc, q_scale, p_loc, p_scale):
        return 0.5 * (
            ((q_scale / p_scale) ** 2)
            + (((q_loc - p_loc) / p_scale) ** 2)
            - 1.0
            + 2.0 * (torch.log(p_scale) - torch.log(q_scale))
        ).sum(dim=-1)

    @staticmethod
    def conditional_negative_elbo(
        log_likelihood,
        latent_kl,
        input_kl,
        beta_kl=1.0,
        beta_denoise=0.1,
    ):
        return (-log_likelihood + beta_kl * latent_kl + beta_denoise * input_kl).mean()

    def _validate_episode(self, x_context, y_context, x_target, y_target):
        tensors = [("x_context", x_context), ("y_context", y_context),
                   ("x_target", x_target)]
        if y_target is not None:
            tensors.append(("y_target", y_target))
        for name, value in tensors:
            if not torch.is_tensor(value) or value.ndim != 3:
                raise ValueError(f"{name} must be a rank-three floating-point tensor")
            if not value.is_floating_point():
                raise ValueError(f"{name} must be a rank-three floating-point tensor")
            if value.shape[0] == 0 or value.shape[1] == 0 or value.shape[2] == 0:
                raise ValueError(f"{name} batch, point, and feature dimensions must be positive")
        if any(value.shape[0] != x_context.shape[0] for _, value in tensors):
            raise ValueError("episode batch dimensions must match")
        if any(value.device != x_context.device or value.dtype != x_context.dtype
               for _, value in tensors):
            raise ValueError("episode tensors must share a device and dtype")
        if x_context.shape[-1] != self.x_dim:
            raise ValueError("x_context last dimension must match x_dim")
        if x_target.shape[-1] != self.x_dim:
            raise ValueError("x_target last dimension must match x_dim")
        if y_context.shape[-1] != self.y_dim:
            raise ValueError("y_context last dimension must match y_dim")
        if x_context.shape[1] != y_context.shape[1]:
            raise ValueError("context point dimensions must match")
        if y_target is not None:
            if y_target.shape[-1] != self.y_dim:
                raise ValueError("y_target last dimension must match y_dim")
            if x_target.shape[1] != y_target.shape[1]:
                raise ValueError("target point dimensions must match")

    def _denoise(self, x):
        batch_size, num_points, _ = x.shape
        hidden = self.denoiser(x.reshape(batch_size * num_points, self.x_dim))
        delta = self.denoiser_loc(hidden).reshape(batch_size, num_points, self.x_dim)
        loc = torch.clamp(x + delta, min=self.x_bounds[0], max=self.x_bounds[1])
        log_scale = self.denoiser_log_scale(hidden).clamp(min=-10.0, max=2.0)
        scale = log_scale.exp().reshape(batch_size, num_points, self.x_dim)
        return loc, scale

    def _encode(self, x, y):
        batch_size, num_points, _ = x.shape
        pair = torch.cat((x, y), dim=-1)
        representation = self.encoder(pair.reshape(batch_size * num_points, -1))
        return representation.reshape(batch_size, num_points, self.r_dim)

    def _latent_distribution(self, pooled):
        hidden = F.relu(self.latent_hidden(pooled))
        loc = self.latent_mean(hidden)
        scale = 0.1 + 0.9 * torch.sigmoid(self.latent_scale(hidden))
        return loc, scale

    def _decode(self, x, z):
        batch_size, num_points, _ = x.shape
        z_expanded = z.unsqueeze(1).expand(-1, num_points, -1)
        hidden = self.decoder(torch.cat((x, z_expanded), dim=-1).reshape(
            batch_size * num_points, self.x_dim + self.z_dim
        ))
        loc = self.output_loc(hidden).reshape(batch_size, num_points, self.y_dim)
        scale = 0.1 + 0.9 * F.softplus(self.output_scale(hidden))
        return loc, scale.reshape(batch_size, num_points, self.y_dim)

    def forward(self, x_context, y_context, x_target, y_target=None, sample=None):
        self._validate_episode(x_context, y_context, x_target, y_target)
        if sample is not None and not isinstance(sample, bool):
            raise ValueError("sample must be a boolean or None")
        sample = self.training if sample is None else sample
        x_observed = torch.cat((x_context, x_target), dim=1)
        x_loc, x_scale = self._denoise(x_observed)
        x_clean = x_loc + x_scale * torch.randn_like(x_scale) if sample else x_loc
        context_size = x_context.shape[1]
        x_clean_context = x_clean[:, :context_size]
        representations = self._encode(x_clean, torch.cat((
            y_context,
            y_target if y_target is not None else y_context.new_zeros(
                x_target.shape[0], x_target.shape[1], self.y_dim
            ),
        ), dim=1)) if y_target is not None else None
        context_representations = self._encode(x_clean_context, y_context)
        context_pooled = context_representations.mean(dim=1)
        z_context_loc, z_context_scale = self._latent_distribution(context_pooled)
        if y_target is not None:
            pooled_representation = representations.mean(dim=1)
            z_all_loc, z_all_scale = self._latent_distribution(pooled_representation)
            z_loc, z_scale = z_all_loc, z_all_scale
        else:
            pooled_representation = context_pooled
            z_all_loc, z_all_scale = None, None
            z_loc, z_scale = z_context_loc, z_context_scale
        z = z_loc + z_scale * torch.randn_like(z_scale) if sample else z_loc
        prediction_loc, prediction_scale = self._decode(x_clean[:, context_size:], z)
        output = {
            "x_posterior_loc": x_loc,
            "x_posterior_scale": x_scale,
            "representations": (representations if representations is not None
                                else context_representations),
            "pooled_representation": pooled_representation,
            "context_pooled_representation": context_pooled,
            "z_context_loc": z_context_loc,
            "z_context_scale": z_context_scale,
            "z_all_loc": z_all_loc,
            "z_all_scale": z_all_scale,
            "prediction_loc": prediction_loc,
            "prediction_scale": prediction_scale,
        }
        if y_target is not None:
            log_likelihood = torch.distributions.Normal(
                prediction_loc, prediction_scale
            ).log_prob(y_target).sum(dim=(1, 2))
            latent_kl = self.diagonal_gaussian_kl(
                z_all_loc, z_all_scale, z_context_loc, z_context_scale
            )
            input_kl = self.diagonal_gaussian_kl(
                x_loc, x_scale, x_observed, self.x_noise_std.view(1, 1, self.x_dim)
            ).sum(dim=1)
            loss = self.conditional_negative_elbo(
                log_likelihood,
                latent_kl,
                input_kl,
                beta_kl=self.beta_kl,
                beta_denoise=self.beta_denoise,
            )
            output.update({
                "log_likelihood": log_likelihood.mean(),
                "latent_kl": latent_kl.mean(),
                "input_kl": input_kl.mean(),
                "loss": loss,
            })
        return output
